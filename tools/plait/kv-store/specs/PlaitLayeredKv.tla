---- MODULE PlaitLayeredKv ----
EXTENDS Naturals

Heads == {0, 1}
Slots == 0..5
Writers == {1, 2}
Select(heads, layers) == {i \in Slots : i \div 3 \in heads /\ i % 3 \in layers}
Legacy == {Select({h}, {0, 1, 2}) : h \in Heads}
ProducerStages == {Select(Heads, {0}), Select(Heads, {1, 2})}
DecoderStages == {Select(Heads, {0, 1}), Select(Heads, {2})}
Writes(w) == IF w = 1 THEN ProducerStages ELSE Legacy
Reads == Legacy \cup DecoderStages
Empty == [i \in Slots |-> <<0, i>>]

VARIABLES attempt, slots, published, receipts
vars == <<attempt, slots, published, receipts>>

Init == attempt = 1 /\ slots = Empty /\ published = FALSE /\ receipts = {}

Write(w, covered) ==
    /\ ~published /\ w = attempt
    /\ slots' = [i \in Slots |-> IF i \in covered THEN <<w, i>> ELSE slots[i]]
    /\ published' = (\A i \in Slots : slots'[i][1] # 0)
    /\ UNCHANGED <<attempt, receipts>>

Expire ==
    /\ ~published /\ attempt = 1 /\ slots # Empty
    /\ attempt' = 2 /\ slots' = Empty
    /\ UNCHANGED <<published, receipts>>

Read(request) ==
    /\ published
    /\ receipts' = receipts \cup {[requested |-> request,
                                 returned |-> [i \in request |-> slots[i]]]}
    /\ UNCHANGED <<attempt, slots, published>>

Next == Expire
        \/ \E w \in Writers : \E covered \in Writes(w) : Write(w, covered)
        \/ \E request \in Reads : Read(request)

Spec == Init /\ [][Next]_vars

OneAttempt == \A i \in Slots : slots[i][1] \in {0, attempt}
CompleteBeforePublication == published => \A i \in Slots : slots[i][1] = attempt
CanonicalReads ==
    \A receipt \in receipts :
        /\ published
        /\ DOMAIN receipt.returned = receipt.requested
        /\ \A i \in receipt.requested : receipt.returned[i] = <<attempt, i>>
====
