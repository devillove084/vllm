---- MODULE PlaitCheckpointPeerPlan ----
EXTENDS Integers

\* Independent stores enforce one peer's directional allocation. A restart
\* discards credit and unfinished grants, and the observer may sample any window.
Stores == {"a", "b"}
PeerBurst == 4
PeerRate == 3
Zero == [s \in Stores |-> 0]
Total(values) == values["a"] + values["b"]
Min(a, b) == IF a <= b THEN a ELSE b
VARIABLES rate, burst, credit, pending, sent, time, observing, startTime, startSent
vars == <<rate, burst, credit, pending, sent, time, observing, startTime, startSent>>

Init ==
    /\ rate \in [Stores -> 1..2]
    /\ burst \in [Stores -> 1..3]
    /\ Total(rate) <= PeerRate
    /\ Total(burst) <= PeerBurst
    /\ credit = Zero /\ pending = Zero
    /\ sent = 0 /\ time = 0
    /\ observing = FALSE /\ startTime = 0 /\ startSent = 0

Tick ==
    /\ time < 4
    /\ time' = time + 1
    /\ credit' = [s \in Stores |-> Min(credit[s] + rate[s], burst[s] - pending[s])]
    /\ UNCHANGED <<rate, burst, pending, sent, observing, startTime, startSent>>

Grant(s) ==
    /\ credit[s] > 0
    /\ credit' = [credit EXCEPT ![s] = @ - 1]
    /\ pending' = [pending EXCEPT ![s] = @ + 1]
    /\ UNCHANGED <<rate, burst, sent, time, observing, startTime, startSent>>

Send(s) ==
    /\ pending[s] > 0
    /\ pending' = [pending EXCEPT ![s] = @ - 1]
    /\ sent' = sent + 1
    /\ UNCHANGED <<rate, burst, credit, time, observing, startTime, startSent>>

Restart(s) ==
    /\ credit' = [credit EXCEPT ![s] = 0]
    /\ pending' = [pending EXCEPT ![s] = 0]
    /\ UNCHANGED <<rate, burst, sent, time, observing, startTime, startSent>>

Sample ==
    /\ ~observing
    /\ observing' = TRUE
    /\ startTime' = time /\ startSent' = sent
    /\ UNCHANGED <<rate, burst, credit, pending, sent, time>>

Next == Tick \/ Sample \/ (\E s \in Stores : Grant(s) \/ Send(s) \/ Restart(s))
Spec == Init /\ [][Next]_vars

SharesFit == Total(rate) <= PeerRate /\ Total(burst) <= PeerBurst
WindowFits == observing => sent - startSent <= PeerBurst + PeerRate * (time - startTime)
====
