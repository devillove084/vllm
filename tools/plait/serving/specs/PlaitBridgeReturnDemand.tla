---- MODULE PlaitBridgeReturnDemand ----
EXTENDS Naturals, Sequences

CONSTANTS RequireTwoWindows, ResetOnMiss, BindWindowIdentity

BridgeCapacity == 3
NoProfile == [epoch |-> 0, revision |-> 0, capacity |-> 0]
NoProposal == [key |-> <<0, 0>>, windows |-> 0]

VARIABLES active, epoch, profile, proposal, recent
vars == <<active, epoch, profile, proposal, recent>>

CurrentKey == <<epoch, profile.revision>>

Init ==
    /\ active = "Bridge"
    /\ epoch = 1
    /\ profile = NoProfile
    /\ proposal = NoProposal
    /\ recent = <<>>

Measure(rate) ==
    /\ active = "Bridge" /\ profile.revision < 3
    /\ profile' = [epoch |-> epoch, revision |-> profile.revision + 1,
                   capacity |-> rate]
    /\ UNCHANGED <<active, epoch, proposal, recent>>

ReplaceTarget ==
    /\ active = "Bridge" /\ epoch = 1
    /\ epoch' = 2
    /\ UNCHANGED <<active, profile, proposal, recent>>

Restart ==
    /\ active = "Bridge"
    /\ proposal' = NoProposal /\ recent' = <<>>
    /\ UNCHANGED <<active, epoch, profile>>

Sample(arrivals, ready) ==
    /\ active = "Bridge"
    /\ LET demand == IF arrivals < BridgeCapacity THEN arrivals ELSE BridgeCapacity
           fits == ready /\ profile.epoch = epoch /\ profile.capacity >= demand
           sample == [key |-> CurrentKey, qualifies |-> fits]
           same == ~BindWindowIdentity \/ proposal.key = CurrentKey
       IN /\ recent' = IF Len(recent) < 2 THEN Append(recent, sample)
                       ELSE <<recent[2], sample>>
          /\ proposal' =
                 IF fits
                 THEN [key |-> CurrentKey,
                       windows |-> IF ~same THEN 1
                                   ELSE IF proposal.windows < 2
                                        THEN proposal.windows + 1 ELSE 2]
                 ELSE IF ResetOnMiss THEN NoProposal ELSE proposal
    /\ UNCHANGED <<active, epoch, profile>>

Qualifies(sample) == sample.qualifies /\ sample.key = CurrentKey

Return ==
    /\ active = "Bridge"
    /\ Len(recent) > 0 /\ Qualifies(recent[Len(recent)])
    /\ proposal.key = CurrentKey
    /\ (~RequireTwoWindows \/ proposal.windows = 2)
    /\ active' = "Target"
    /\ UNCHANGED <<epoch, profile, proposal, recent>>

Next ==
    \/ \E rate \in 2..3 : Measure(rate)
    \/ ReplaceTarget \/ Restart \/ Return
    \/ \E arrivals \in 1..4, ready \in BOOLEAN : Sample(arrivals, ready)

Spec == Init /\ [][Next]_vars
TwoMatchingWindowsAtReturn ==
    [][active' = "Target" /\ active = "Bridge" =>
       Len(recent) = 2 /\ \A i \in 1..2 : Qualifies(recent[i])]_vars
====
