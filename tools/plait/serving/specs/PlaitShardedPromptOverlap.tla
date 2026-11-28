---- MODULE PlaitShardedPromptOverlap ----
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS KeepBlockOrder, SlotCount

Slots == 1..SlotCount
Blocks == 1..3
Call(block, kind) == [block |-> block, kind |-> kind]
\* The packet determines each block's branch. Exercise partial, empty,
\* and full dense fallback in one request.
Calls == <<Call(1, "summary"), Call(1, "rank"), Call(1, "dense"),
           Call(2, "summary"), Call(2, "rank"),
           Call(3, "summary"), Call(3, "dense")>>
EndOfWire == Len(Calls)
Idle == [seq |-> 0, async |-> FALSE]

VARIABLES prepared, consumed, outstanding, ready
vars == <<prepared, consumed, outstanding, ready>>

Init ==
    /\ prepared = [slot \in Slots |-> {}]
    /\ consumed = [slot \in Slots |-> 0]
    /\ outstanding = [slot \in Slots |-> Idle]
    /\ ready = 0

Prepare(slot, block) ==
    /\ consumed[slot] < EndOfWire
    /\ block \notin prepared[slot]
    /\ LET current == Calls[consumed[slot] + 1]
       IN \/ block = current.block
          \/ /\ block = current.block + 1
             /\ current.kind = "summary"
             /\ outstanding[slot].async
    /\ prepared' = [prepared EXCEPT ![slot] = @ \cup {block}]
    /\ UNCHANGED <<consumed, outstanding, ready>>

Issue(slot, seq, async) ==
    /\ outstanding[slot] = Idle /\ consumed[slot] < EndOfWire
    /\ Calls[seq].block \in prepared[slot]
    /\ \/ seq = consumed[slot] + 1
       \/ /\ ~KeepBlockOrder
          /\ Calls[seq] = Call(Calls[consumed[slot] + 1].block + 1, "summary")
    /\ async => \/ (Calls[seq].kind = "summary" /\ Calls[seq].block < 3)
                \/ Calls[seq].kind = "rank"
    /\ outstanding' = [outstanding EXCEPT ![slot] = [seq |-> seq, async |-> async]]
    /\ UNCHANGED <<prepared, consumed, ready>>

FinishWire ==
    /\ ready < EndOfWire
    /\ \A slot \in Slots : outstanding[slot].seq = ready + 1
    /\ ready' = ready + 1
    /\ UNCHANGED <<prepared, consumed, outstanding>>

Consume(slot) ==
    /\ LET seq == outstanding[slot].seq
       IN /\ 0 < seq /\ seq <= ready
          /\ consumed' = [consumed EXCEPT ![slot] = seq]
          /\ prepared' = IF seq = EndOfWire \/ Calls[seq].block # Calls[seq + 1].block
                         THEN [prepared EXCEPT ![slot] = @ \ {Calls[seq].block}]
                         ELSE prepared
    /\ outstanding' = [outstanding EXCEPT ![slot] = Idle]
    /\ UNCHANGED ready

Next ==
    \/ FinishWire
    \/ \E slot \in Slots :
        \/ Consume(slot)
        \/ \E block \in Blocks : Prepare(slot, block)
        \/ \E seq \in 1..EndOfWire, async \in BOOLEAN : Issue(slot, seq, async)

Spec == Init /\ [][Next]_vars

CollectiveOrder == \A slot \in Slots :
    outstanding[slot] # Idle => outstanding[slot].seq = consumed[slot] + 1

ProjectionBound == \A slot \in Slots : Cardinality(prepared[slot]) <= 2

RequestCanAdvance ==
    (\A slot \in Slots : consumed[slot] = EndOfWire) \/ ENABLED Next
====
