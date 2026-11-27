---- MODULE PlaitPromptPlan ----
EXTENDS Naturals

CONSTANT AgreeBeforeSelecting

Slots == 1..2
\* 0 means no usable shard. 1 and 2 represent different valid geometries.
Offers == [geometry : 0..2, denseSubmit : BOOLEAN, overlap : BOOLEAN]
Plans == {plan \in [sharded : BOOLEAN, denseOverlap : BOOLEAN, shardOverlap : BOOLEAN] :
             plan.shardOverlap => plan.sharded}
NoOffer == [geometry |-> 3, denseSubmit |-> FALSE, overlap |-> FALSE]
NoPlan == [sharded |-> FALSE, denseOverlap |-> FALSE, shardOverlap |-> TRUE]

VARIABLES offers, plans, issued
vars == <<offers, plans, issued>>

Init ==
    /\ offers = [slot \in Slots |-> NoOffer]
    /\ plans = [slot \in Slots |-> NoPlan]
    /\ issued = {}

EveryoneOffered == \A slot \in Slots : offers[slot] # NoOffer

Choose(members) ==
    LET compatible == \E geometry \in 1..2 :
                         \A slot \in members : offers[slot].geometry = geometry
    IN [sharded |-> compatible,
        denseOverlap |-> \A slot \in members : offers[slot].denseSubmit,
        shardOverlap |-> compatible /\
                         (\A slot \in members : offers[slot].overlap)]

Declare(slot, offer) ==
    /\ offers[slot] = NoOffer
    /\ offers' = [offers EXCEPT ![slot] = offer]
    /\ UNCHANGED <<plans, issued>>

Select(slot) ==
    /\ plans[slot] = NoPlan /\ offers[slot] # NoOffer
    /\ IF AgreeBeforeSelecting THEN EveryoneOffered ELSE TRUE
    /\ LET members == IF AgreeBeforeSelecting THEN Slots ELSE {slot}
       IN plans' = [plans EXCEPT ![slot] = Choose(members)]
    /\ UNCHANGED <<offers, issued>>

IssueFirstCollective(slot) ==
    /\ slot \notin issued /\ plans[slot] # NoPlan
    /\ issued' = issued \cup {slot}
    /\ UNCHANGED <<offers, plans>>

Next ==
    \E slot \in Slots :
        \/ \E offer \in Offers : Declare(slot, offer)
        \/ Select(slot)
        \/ IssueFirstCollective(slot)

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ offers \in [Slots -> Offers \cup {NoOffer}]
    /\ plans \in [Slots -> Plans \cup {NoPlan}]
    /\ issued \subseteq Slots

CommonPlan ==
    \A first, second \in Slots :
        (plans[first] # NoPlan /\ plans[second] # NoPlan) =>
            plans[first] = plans[second]

UseOnlyCommonCapabilities ==
    \A slot \in Slots :
        plans[slot] # NoPlan => EveryoneOffered /\ plans[slot] = Choose(Slots)

WireKind(slot) == IF plans[slot].sharded THEN "summary" ELSE "dense"
SameFirstCollective ==
    \A first, second \in issued : WireKind(first) = WireKind(second)

====
