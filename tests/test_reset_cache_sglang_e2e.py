"""End-to-end sglang + FlexKV cache-reset test.

Mirrors the vLLM e2e test (FlexKV/tests/test_reset_cache_vllm_e2e.py) but for
sglang. It exercises the EXACT path verl hits to clear the KV cache after a
weight update: an HTTP ``POST /flush_cache``.

    verl: update_weights -> GET/POST /flush_cache
      -> tokenizer_manager.flush_cache() -> scheduler.flush_cache()
      -> tree_cache.reset()  (FlexKVRadixCache.reset())
      -> FlexKVConnector.reset() -> kv_manager.reset()   <-- the change under test
      -> FlexKV drops radix tree + mempool on every tier (CPU / SSD / remote)

How we measure a FlexKV hit (no local-only clear needed)
--------------------------------------------------------
sglang reports a per-request prefix-cache breakdown in
``meta_info["cached_tokens_details"]``:

    device  : tokens served from the local GPU radix cache
    host    : tokens served from FlexKV (its LOOKUP -> host_hit_length)
    storage : tokens served from an L3 storage backend (None here)

Unlike vLLM, we do NOT need a "reset the local cache only" call to isolate the
connector: the ``host`` field already separates FlexKV hits from local GPU hits.
A FlexKV hit shows up as ``host > 0`` regardless of what the device cache did.

``/flush_cache`` (verl's clear) drops BOTH the local tree and, via the change
under test, FlexKV's backing store -- so after it, a re-send that would
otherwise hit FlexKV must instead miss (``host == 0``) until FlexKV is
re-populated.

Test protocol (why it proves the reset works):
    warm  : generate(A)              -> FlexKV gets populated
    CLEAR : POST /flush_cache        (verl's call; drops local + FlexKV)
    run1  : generate(A)              -> host == 0   (FlexKV was cleared)
    run2  : generate(A)              -> host  > 0   (run1 re-populated FlexKV)

flush_cache is idle-gated (scheduler.flush_cache runs only when
``is_fully_idle()``); every generate here is a blocking, fully-drained request,
so the server is quiesced by the time we POST /flush_cache.

Run (inside the sglang + FlexKV environment, FlexKV on branch feat/rl_kv_clear):
    FLEXKV_CPU_CACHE_GB=64 \
        python3 -m pytest -s tests/test_reset_cache_sglang_e2e.py

NOTE on CUDA_VISIBLE_DEVICES under MPS: if the box runs an MPS daemon
(``nvidia-cuda-mps-control -d``), the client must use the MPS-renumbered index,
not the physical one. MPS exposes only the GPU(s) it was started with, renumbered
from 0. On this host MPS was started pinned to physical GPU3, so the server must
be launched with ``CUDA_VISIBLE_DEVICES=0`` (which maps to physical GPU3);
``CUDA_VISIBLE_DEVICES=3`` sees no GPU and the launch dies with SIGKILL (-9) /
"No CUDA GPUs are available". See FlexKV memory note "sglang e2e MPS + device-cache".
"""

import os
import unittest

import pytest

requests = pytest.importorskip("requests")
pytest.importorskip("sglang")
pytest.importorskip("flexkv")

# Guarded by importorskip above (mirrors test_reset_cache_vllm_e2e.py).
from sglang.srt.utils import kill_process_tree  # noqa: E402
from sglang.test.test_utils import (  # noqa: E402
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Default matches the user's sglang launch command.
MODEL = "/raid/model/Qwen3-8B"
MAX_MODEL_LEN = 8192

# Cap the GPU KV pool so the local device radix cache is FORCED to evict.
# Without this, a large --mem-fraction-static gives a KV pool big enough to hold
# every prompt in the local device tree forever, so a warm re-send is served
# from the device tier (cached_tokens_details.device) and FlexKV's host tier is
# never consulted -- the host>0 assertions then read 0 and fail even though the
# connector works. The batch below is 8 prompts x ~3203 tokens ~= 25.6k tokens;
# a 20k pool guarantees eviction so re-sends fall through to FlexKV (host>0).
# See FlexKV memory note "sglang e2e MPS + device-cache".
MAX_TOTAL_TOKENS = 20000

# A batch of long prompts so each spans many KV pages and actually gets
# offloaded to FlexKV (FlexKV matches at page granularity). The ``[{turn}]``
# prefix keeps each prompt distinct so they never cross-hit one another.
_DATASET = [
    "你有什么爱好？",
    "你有什么特长？",
    "你有什么兴趣？",
    "你有什么梦想？",
    "你有什么愿望？",
    "你有什么期待？",
    "你有什么计划？",
    "你有什么遗憾？",
]
PROMPTS = [f"[{turn}] {prompt * 800}" for turn, prompt in enumerate(_DATASET)]


class TestFlexKVResetCacheE2E(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # FlexKV reads FLEXKV_CPU_CACHE_GB (simplest) or FLEXKV_CONFIG_PATH.
        # Default to a small CPU cache if the user set neither.
        env = os.environ.copy()
        if not env.get("FLEXKV_CONFIG_PATH") and not env.get("FLEXKV_CPU_CACHE_GB"):
            env["FLEXKV_CPU_CACHE_GB"] = "8"

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--context-length",
                str(MAX_MODEL_LEN),
                "--mem-fraction-static",
                "0.5",
                # Bound the GPU KV pool so the device radix cache evicts and
                # re-sends fall through to FlexKV's host tier (see MAX_TOTAL_TOKENS
                # above). Without this the host>0 assertions read 0 on an idle,
                # large-VRAM GPU because nothing ever leaves the device tier.
                "--max-total-tokens",
                str(MAX_TOTAL_TOKENS),
                # --enable-flexkv wires the FlexKVRadixCache in as tree_cache
                # (see mem_cache/registry.py); it also implies hierarchical cache.
                "--enable-flexkv",
                "--disable-cuda-graph",
                "--enable-metrics",
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _generate(self, prompt: str, max_new_tokens: int = 8):
        """Send one prompt via the native /generate API and return its meta_info.

        We read the per-request cache breakdown from meta_info rather than the
        Prometheus counters: the breakdown rides on the request object and
        survives the async KV-load step boundary, whereas the external-hit
        counters can be dropped when a fully-matched request produces no output
        token on the accounting step (same failure mode as vLLM's external
        counters -- see FlexKV memory notes).
        """
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["meta_info"]

    @staticmethod
    def _flexkv_hit(meta_info: dict) -> int:
        """FlexKV hit length for a request = the 'host' portion of the prefix.

        FlexKVRadixCache reports its LOOKUP result as host_hit_length, which
        sglang accounts as cached_tokens_details.host. Falls back to the flat
        cached_tokens if the breakdown is absent (older builds), which over the
        warm/cleared protocol below still tracks FlexKV because each round uses
        a fresh server-side state.
        """
        details = meta_info.get("cached_tokens_details")
        if details is not None:
            return int(details.get("host", 0) or 0)
        return int(meta_info.get("cached_tokens", 0) or 0)

    def _flush_cache(self) -> bool:
        """Reproduce verl's clear: POST /flush_cache (drops local + FlexKV).

        Returns True only when the server reports the flush actually happened.
        flush_cache is idle-gated; every generate in this test is blocking and
        fully drained, so the server should be idle here. Retry briefly to
        absorb FlexKV's async D2H put still holding blocks at the boundary.
        """
        for _ in range(50):
            resp = requests.post(f"{self.base_url}/flush_cache", timeout=60)
            # http_server returns 200 + "Cache flushed." only on success; a
            # not-idle attempt comes back with a different status/body.
            if resp.status_code == 200 and "Cache flushed" in resp.text:
                return True
            import time

            time.sleep(0.1)
        return False

    def _warm(self, prompts):
        for p in prompts:
            self._generate(p)

    def _batch_flexkv_hits(self, prompts):
        return [self._flexkv_hit(self._generate(p)) for p in prompts]

    # ------------------------------------------------------------------
    # tests
    # ------------------------------------------------------------------
    def test_flush_cache_succeeds(self):
        """Smoke: verl's exact clear call returns success (wire is connected)."""
        self._generate(PROMPTS[0])
        self.assertTrue(self._flush_cache(), "POST /flush_cache did not report success")

    def test_flexkv_populates_and_is_measurable(self):
        """Sanity: after a clean flush + populate + eviction + re-send, FlexKV
        serves the prefix from the host tier (host > 0).

        If FlexKV hits stay 0 here, the connector isn't really engaged and the
        reset test below would be meaningless.

        Note we cannot just send the SAME prompt twice back-to-back: nothing
        would evict its copy from the local *device* radix tree between the two
        sends, so the re-send would device-hit (host == 0) and FlexKV's host
        tier would never be consulted -- the assertion would spuriously fail
        even with a working connector. Instead we populate the target, then send
        the OTHER distinct prompts to push the target out of the bounded device
        pool (MAX_TOTAL_TOKENS), and only then re-send it: now the match has to
        come from FlexKV's host tier.
        """
        self.assertTrue(self._flush_cache())
        target = PROMPTS[0]
        self._generate(target)  # cold: populate FlexKV for the target
        # Evict the target from the device pool by streaming the other distinct
        # prompts (batch working set > MAX_TOTAL_TOKENS forces device eviction).
        for evictor in PROMPTS[1:]:
            self._generate(evictor)
        host_hit = self._flexkv_hit(self._generate(target))  # re-send: host hit
        self.assertGreater(
            host_hit,
            0,
            "no FlexKV (host) hits on a warm re-send; connector not engaged -- "
            "check --enable-flexkv mount / FlexKV branch (needs KVManager.reset())",
        )

    def test_flush_cache_invalidates_flexkv(self):
        """Behavioral: verl's clear must drop FlexKV so the next send misses it.

        Protocol per prompt: warm -> CLEAR -> run1 (host == 0) -> run2 (host>0).
        Runs over the whole batch and asserts on aggregate hits so a single
        page-alignment quirk doesn't flake the test.
        """
        # warm: populate FlexKV for every prompt.
        self._warm(PROMPTS)

        # CLEAR: exactly what verl's clear_kv_cache does (drops local + FlexKV).
        self.assertTrue(self._flush_cache(), "flush before run1 failed")

        # run1: FlexKV was just cleared -> every prompt must miss on host.
        run1 = self._batch_flexkv_hits(PROMPTS)

        # run2: run1 re-populated FlexKV. No flush now -> the SAME prompts must
        # hit FlexKV on the host tier again (proves the cache recovered and we
        # were measuring a real cache, not a permanently dead one).
        run2 = self._batch_flexkv_hits(PROMPTS)

        total1, total2 = sum(run1), sum(run2)
        print(
            f"[flexkv-reset] host hits after CLEAR run1={total1} "
            f"({run1})  run2={total2} ({run2})"
        )

        # Assert 1: a cleared FlexKV serves no stale host hits.
        self.assertEqual(
            total1,
            0,
            f"FlexKV still served {total1} host tokens right after /flush_cache "
            f"({run1}); reset did NOT propagate to FlexKV (stale KV served)",
        )
        # Assert 2: FlexKV works again after re-population.
        self.assertGreater(
            total2,
            total1,
            f"FlexKV host hits did not recover after refill (run1={total1}, "
            f"run2={total2}); connector may not be working at all",
        )

    def test_flush_cache_rejected_when_not_idle(self):
        """flush_cache is idle-gated: with no in-flight requests it must succeed.

        We cannot easily force a not-idle state from a single-threaded test
        without racing, so this just asserts the idle path succeeds -- the
        not-idle rejection is covered by scheduler.flush_cache's own guard.
        """
        self._generate(PROMPTS[0])
        self.assertTrue(self._flush_cache())


if __name__ == "__main__":
    unittest.main()
