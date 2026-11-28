---- MODULE PlaitPipelineCacheBudget ----
\* Two TP workers per stage. Host 2 starts at global rank 1, so its
\* local worker ordinal cannot identify the pipeline stage.
EXTENDS Naturals

CONSTANTS RequireAgreement, UseGlobalRank
Hosts == 1..2
Ranks == 0..3
Profiles == 1..3
Layers == <<18, 10>>

\* Profiles 1 and 2 have the same budget but different launch descriptors:
\* the first uses the scalar default and the second transmits a vector.
Budgets == << <<256, 256>>, <<256, 256>>, <<329, 183>> >>
Host(rank) == IF rank = 0 THEN 1 ELSE 2
Stage(rank) == 1 + (rank \div 2)
LocalRank(rank) == IF Host(rank) = 1 THEN rank ELSE rank - 1

VARIABLES offered, agreed, allocated
vars == <<offered, agreed, allocated>>

Init ==
    /\ offered = [host \in Hosts |-> 0]
    /\ agreed = FALSE
    /\ allocated = [rank \in Ranks |-> 0]

Offer(host, profile) ==
    /\ offered[host] = 0
    /\ offered' = [offered EXCEPT ![host] = profile]
    /\ UNCHANGED <<agreed, allocated>>

Confirm ==
    /\ ~agreed
    /\ \A host \in Hosts : offered[host] \in Profiles
    /\ RequireAgreement => offered[1] = offered[2]
    /\ agreed' = TRUE
    /\ UNCHANGED <<offered, allocated>>

Allocate(rank) ==
    /\ agreed /\ allocated[rank] = 0
    /\ LET index == IF UseGlobalRank THEN rank ELSE LocalRank(rank)
       IN allocated' = [allocated EXCEPT ![rank] = Budgets[offered[Host(rank)]][Stage(index)]]
    /\ UNCHANGED <<offered, agreed>>

Next ==
    \/ \E host \in Hosts, profile \in Profiles : Offer(host, profile)
    \/ Confirm
    \/ \E rank \in Ranks : Allocate(rank)
Spec == Init /\ [][Next]_vars

SameBudgetPlan == agreed => offered[1] = offered[2]
WorkerUsesStageBudget ==
    \A rank \in Ranks : allocated[rank] # 0 =>
        allocated[rank] = Budgets[offered[Host(rank)]][Stage(rank)]

Minimum(values) == CHOOSE x \in values : \A y \in values : x <= y
\* A 128-token block uses 0.5 MiB per layer of FP16 K and V.
LogicalTokens ==
    128 * Minimum({(2 * allocated[rank]) \div Layers[Stage(rank)] : rank \in Ranks})
CapacityMatchesAllocation ==
    (\A rank \in Ranks : allocated[rank] # 0) =>
        LogicalTokens = IF offered[1] = 3 THEN 4608 ELSE 3584
====
