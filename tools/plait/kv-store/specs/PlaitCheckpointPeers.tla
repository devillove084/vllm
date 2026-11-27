---- MODULE PlaitCheckpointPeers ----
EXTENDS Integers

Peers == {"fast", "slow"}
ReadCap == [p \in Peers |-> IF p = "fast" THEN 4 ELSE 2]
WriteCap == [p \in Peers |-> IF p = "fast" THEN 3 ELSE 1]
GlobalCap == 6
Zero == [p \in Peers |-> 0]
Total(x) == x["fast"] + x["slow"]
Min(a, b) == IF a <= b THEN a ELSE b
VARIABLES readCredit, writeCredit, globalCredit, leased, reading, writing, time
vars == <<readCredit, writeCredit, globalCredit, leased, reading, writing, time>>

Init ==
    /\ readCredit = ReadCap
    /\ writeCredit = WriteCap
    /\ globalCredit = GlobalCap
    /\ leased = Zero /\ reading = Zero /\ writing = Zero
    /\ time = 0

Pending == Total(leased) + Total(reading) + Total(writing)
Refill ==
    /\ time < 3
    /\ time' = time + 1
    /\ globalCredit' = Min(globalCredit + 1, GlobalCap - Pending)
    /\ readCredit' = [p \in Peers |-> Min(readCredit[p] + 1, ReadCap[p] - leased[p] - reading[p])]
    /\ writeCredit' = [p \in Peers |-> Min(writeCredit[p] + 1, WriteCap[p] - writing[p])]
    /\ UNCHANGED <<leased, reading, writing>>

Reserve(shards) ==
    /\ Total(shards) <= globalCredit
    /\ \A p \in Peers : shards[p] <= readCredit[p]
    /\ leased' = [p \in Peers |-> leased[p] + shards[p]]
    /\ globalCredit' = globalCredit - Total(shards)
    /\ readCredit' = [p \in Peers |-> readCredit[p] - shards[p]]
    /\ UNCHANGED <<writeCredit, reading, writing, time>>

Claim(p) ==
    /\ leased[p] > 0
    /\ leased' = [leased EXCEPT ![p] = @ - 1]
    /\ reading' = [reading EXCEPT ![p] = @ + 1]
    /\ UNCHANGED <<readCredit, writeCredit, globalCredit, writing, time>>

AdmitWrite(p) ==
    /\ globalCredit > 0 /\ writeCredit[p] > 0
    /\ globalCredit' = globalCredit - 1
    /\ writeCredit' = [writeCredit EXCEPT ![p] = @ - 1]
    /\ writing' = [writing EXCEPT ![p] = @ + 1]
    /\ UNCHANGED <<readCredit, leased, reading, time>>

FinishRead(p) ==
    /\ reading[p] > 0
    /\ reading' = [reading EXCEPT ![p] = @ - 1]
    /\ UNCHANGED <<readCredit, writeCredit, globalCredit, leased, writing, time>>

FinishWrite(p) ==
    /\ writing[p] > 0
    /\ writing' = [writing EXCEPT ![p] = @ - 1]
    /\ UNCHANGED <<readCredit, writeCredit, globalCredit, leased, reading, time>>

Expire(p) ==
    /\ leased[p] > 0
    /\ leased' = [leased EXCEPT ![p] = 0]
    /\ UNCHANGED <<readCredit, writeCredit, globalCredit, reading, writing, time>>

Next == Refill \/ (\E shards \in [Peers -> 1..3] : Reserve(shards))
    \/ (\E p \in Peers : Claim(p) \/ AdmitWrite(p) \/ FinishRead(p) \/ FinishWrite(p) \/ Expire(p))
Spec == Init /\ [][Next]_vars

Nonnegative == /\ globalCredit >= 0
    /\ \A p \in Peers : /\ readCredit[p] >= 0 /\ writeCredit[p] >= 0
                       /\ leased[p] >= 0 /\ reading[p] >= 0 /\ writing[p] >= 0
GlobalBudget == globalCredit + Pending <= GlobalCap
PeerBudgets == \A p \in Peers :
    /\ readCredit[p] + leased[p] + reading[p] <= ReadCap[p]
    /\ writeCredit[p] + writing[p] <= WriteCap[p]
====
