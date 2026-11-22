"""System-2 client: P(Yes) parsing, load shedding, deadline, escalation log.
Runs on the virtual clock (bench/controlled/vclock.py): deterministic, instant."""
import asyncio, json, math, os, sys, tempfile
sys.path[:0] = ["mitmproxy", "bench/controlled"]
import httpx
import system2
import vclock

def _choice(yes):
    return {"message": {"content": "Yes" if yes > 0.5 else "No"},
            "logprobs": {"content": [{"top_logprobs": [{"token": "▁Yes", "logprob": math.log(yes)},
                                                       {"token": "No", "logprob": math.log(1 - yes)}]}]}}

def test_p_yes_parsing():
    assert abs(system2.p_yes(_choice(0.8)) - 0.8) < 1e-9
    assert system2.p_yes({"message": {"content": "Yes."}}) == 0.9          # no logprobs: text fallback
    only_no = {"logprobs": {"content": [{"top_logprobs": [{"token": "No", "logprob": math.log(0.7)}]}]}}
    assert abs(system2.p_yes(only_no) - 0.3) < 1e-9

def _judge_transport(delay_s, yes=0.9):
    async def h(req):
        await asyncio.sleep(delay_s)
        return httpx.Response(200, json={"choices": [_choice(yes)]})
    return httpx.MockTransport(h)

def test_shed_timeout_and_log():
    with tempfile.TemporaryDirectory() as d:
        system2.ESCALATION_LOG = os.path.join(d, "esc.jsonl")
        system2.configure(mode="inline", transport=_judge_transport(0.4), url="http://j",
                          timeout_ms=1000, max_concurrency=2, max_queue=2)
        async def burst():
            return await asyncio.gather(*(system2.judge("tool", "injection", f"t{i}", s1=0.5, key=str(i))
                                          for i in range(6)))
        vs = vclock.run(burst())
        st = [v.status for v in vs]
        # 2 run at once, 2 queue, 2 shed immediately; the queued pair finishes at 0.8 s < 1.0 s deadline
        assert st.count("shed") == 2 and st.count("ok") == 4, st
        assert all(v.block for v in vs if v.status == "ok")
        # deadline: a judge slower than S2_TIMEOUT_MS falls back (p=None), never raises
        system2.configure(mode="inline", transport=_judge_transport(2.0), timeout_ms=1000)
        v = vclock.run(system2.judge("user", "injection", "x"))
        assert v.status == "timeout" and v.p is None and v.block is None and abs(v.ms - 1000) < 1e-6
        recs = [json.loads(l) for l in open(system2.ESCALATION_LOG)]
        assert len(recs) == 7 and {r["status"] for r in recs} == {"ok", "shed", "timeout"}
        assert "text" not in recs[0]                                    # no content unless opted in
        system2.ESCALATION_LOG = None
    system2.configure(mode="off")
    assert vclock.run(system2.judge("user", "injection", "x")).status == "off"

if __name__ == "__main__":
    test_p_yes_parsing(); test_shed_timeout_and_log()
    print("test_system2: OK")
