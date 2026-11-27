---- MODULE PlaitKvPublication ----
EXTENDS Naturals

Heads == 0..3
Writers == 1..2
TargetShards == {0..1, 2..3}
Value(writer, head) == 10 * writer + head

VARIABLES writer, heads, reads
vars == <<writer, heads, reads>>
Complete == DOMAIN heads = Heads

Init ==
    /\ writer = 0
    /\ heads = [h \in {} |-> 0]
    /\ reads = {}

Deposit(w, span) ==
    /\ ~Complete
    /\ writer \in {0, w}
    /\ span \cap DOMAIN heads = {}
    /\ writer' = w
    /\ heads' = [h \in DOMAIN heads \cup span |->
        IF h \in span THEN Value(w, h) ELSE heads[h]]
    /\ UNCHANGED reads

Expire ==
    /\ writer # 0 /\ ~Complete
    /\ writer' = 0
    /\ heads' = [h \in {} |-> 0]
    /\ UNCHANGED reads

Fetch(span) ==
    /\ Complete
    /\ reads' = reads \cup {
        [range |-> span, covered |-> DOMAIN heads, values |-> [h \in span |-> heads[h]]]}
    /\ UNCHANGED <<writer, heads>>

Next ==
    \/ Expire
    \/ \E w \in Writers, span \in {{0}, {1}, {2}, {3}, 0..1, 2..3} : Deposit(w, span)
    \/ \E span \in TargetShards : Fetch(span)

Spec == Init /\ [][Next]_vars

NoMixedWriters == \A h \in DOMAIN heads : heads[h] = Value(writer, h)
FullCoverageBeforeRead == \A read \in reads : read.covered = Heads
RangeOrderAndIdentity == \A read \in reads :
    /\ DOMAIN read.values = read.range
    /\ \A h \in read.range : read.values[h] = Value(writer, h)
CompleteIsImmutable == [][Complete => heads' = heads /\ writer' = writer]_vars
====
