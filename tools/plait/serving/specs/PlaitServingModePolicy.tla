---- MODULE PlaitServingModePolicy ----
\* One version per directory group. Every snapshot includes the unused groups,
\* represented by absence from its set. A policy submission requires two full
\* intervals for one proposal. The first apply and exact retry share one ID.
\* Preferred abstracts the demand and SLO calculation. The model checks local
\* pre-send evidence and ordered application. The network may delay an apply
\* after its capacity evidence expires or its HTTP probes become stale.
EXTENDS Naturals, Sequences

Aggregate == {1}
Split == {2, 3}
Mixed == Aggregate \cup Split
Spare == 4
Decode == 3
Catalog == Mixed \cup {Spare}
Modes == {Aggregate, Split, Mixed, {Spare}}
Demand == {"PreferSplit", "PreferMixed", "NoGain", "LowerServed"}
Target(demand) == IF demand = "PreferMixed" THEN Mixed ELSE Split
\* NoGain includes positive scores below the configured minimum; LowerServed
\* can have a positive score while serving fewer requests.
PositiveGain(demand) == demand # "NoGain"
PreservesServed(demand) == demand # "LowerServed"
Preferred(demand) == PositiveGain(demand) /\ PreservesServed(demand)

Basis(incumbent, target, revision, demand) ==
    [incumbent |-> incumbent, target |-> target, revision |-> revision, demand |-> demand]
Intent(id, expected, target) == [id |-> id, expected |-> expected, target |-> target]
NoIntent == Intent(0, {}, {})
Pending == "Pending"
Applied == "Applied"
Rejected == "Rejected"

VARIABLES native, profiles, local, request, receipt, repeated
vars == <<native, profiles, local, request, receipt, repeated>>

Init ==
    /\ native = [active |-> Aggregate, ready |-> Catalog \ Aggregate,
                  http |-> Catalog \ {Decode}, disturbed |-> FALSE]
    /\ profiles = Modes
    /\ local = [revision |-> 1, observed |-> {}, read |-> FALSE, age |-> "Unseeded", support |-> <<>>]
    /\ request = NoIntent /\ receipt = Pending /\ repeated = FALSE

Observe ==
    /\ request = NoIntent /\ ~local.read
    /\ local' = [local EXCEPT !.observed = native.active, !.read = TRUE]
    /\ UNCHANGED <<native, profiles, request, receipt, repeated>>

Seed ==
    /\ request = NoIntent /\ local.read /\ local.age = "Unseeded"
    /\ local' = [local EXCEPT !.age = "Short", !.read = FALSE]
    /\ UNCHANGED <<native, profiles, request, receipt, repeated>>

ShortPoll ==
    /\ request = NoIntent /\ local.read /\ local.age = "Short"
    /\ local' = [local EXCEPT !.read = FALSE]
    /\ UNCHANGED <<native, profiles, request, receipt, repeated>>

Elapsed ==
    /\ request = NoIntent /\ local.age = "Short"
    /\ local' = [local EXCEPT !.age = "Full"]
    /\ UNCHANGED <<native, profiles, request, receipt, repeated>>

ReloadPolicy ==
    /\ request = NoIntent /\ local.revision = 1
    /\ local' = [local EXCEPT !.revision = 2, !.age = "Unseeded", !.read = FALSE, !.support = <<>>]
    /\ UNCHANGED <<native, profiles, request, receipt, repeated>>

ResetSampler ==
    /\ request = NoIntent /\ local.age # "Unseeded"
    /\ local' = [local EXCEPT !.age = "Unseeded", !.read = FALSE, !.support = <<>>]
    /\ UNCHANGED <<native, profiles, request, receipt, repeated>>

CanChoose(demand) ==
    LET incumbent == local.observed
        target == Target(demand)
    IN /\ incumbent \in Modes \cap profiles /\ target \in profiles /\ incumbent # target
       /\ Preferred(demand)
       /\ target \cap incumbent \subseteq native.active
       /\ target \ incumbent \subseteq native.ready
       /\ target \subseteq native.http

SameProposal(a, b) ==
    a.incumbent = b.incumbent /\ a.target = b.target /\ a.revision = b.revision

AddWindow(support, evidence) ==
    IF Len(support) = 0 THEN <<evidence>>
    ELSE IF SameProposal(support[Len(support)], evidence)
         THEN <<support[Len(support)], evidence>> ELSE <<evidence>>

CompleteWindow(demand) ==
    /\ request = NoIntent /\ local.read /\ local.age = "Full"
    /\ LET evidence == Basis(local.observed, Target(demand), local.revision, demand)
       IN local' = [local EXCEPT !.age = "Short", !.read = FALSE,
                     !.support = IF CanChoose(demand) THEN AddWindow(@, evidence) ELSE <<>>]
    /\ UNCHANGED <<native, profiles, request, receipt, repeated>>

\* This is the local check before sending. Its incumbent is the exact mode
\* selected from the last two observations, and its expected covers Catalog.
Send ==
    /\ request = NoIntent /\ Len(local.support) = 2
    /\ LET chosen == local.support[2]
       IN /\ {chosen.incumbent, chosen.target} \subseteq profiles
          /\ chosen.incumbent \subseteq native.active
          /\ request' = Intent(1, chosen.incumbent, chosen.target)
    /\ UNCHANGED <<native, profiles, local, receipt, repeated>>

\* The native operation checks the compare even for unchanged/absent groups.
FirstApply ==
    /\ request # NoIntent /\ receipt = Pending
    /\ LET accepted == native.active = request.expected /\ request.target \ native.active \subseteq native.ready
       IN /\ native' = IF accepted THEN [native EXCEPT !.active = request.target,
                          !.ready = @ \ request.target] ELSE native
          /\ receipt' = IF accepted THEN Applied ELSE Rejected
    /\ UNCHANGED <<profiles, local, request, repeated>>

ExactRetry ==
    /\ request # NoIntent /\ receipt # Pending /\ ~repeated
    /\ repeated' = TRUE
    /\ UNCHANGED <<native, profiles, local, request, receipt>>

Expire(mode) ==
    /\ profiles = Modes /\ mode \in profiles
    /\ profiles' = profiles \ {mode}
    /\ UNCHANGED <<native, local, request, receipt, repeated>>

HttpReady ==
    /\ Decode \notin native.http /\ Decode \in native.ready
    /\ native' = [native EXCEPT !.http = @ \cup {Decode}]
    /\ UNCHANGED <<profiles, local, request, receipt, repeated>>

IndependentSwitch ==
    /\ ~native.disturbed /\ Spare \in native.ready
    /\ native' = [native EXCEPT !.active = @ \cup {Spare}, !.ready = @ \ {Spare}, !.disturbed = TRUE]
    /\ UNCHANGED <<profiles, local, request, receipt, repeated>>

ReportFailure(group) ==
    /\ ~native.disturbed /\ group \in native.active \cup native.ready
    /\ native' = [native EXCEPT !.active = @ \ {group}, !.ready = @ \ {group},
                  !.http = @ \ {group}, !.disturbed = TRUE]
    /\ UNCHANGED <<profiles, local, request, receipt, repeated>>

Next ==
    \/ Observe \/ Seed \/ ShortPoll \/ Elapsed \/ ReloadPolicy \/ ResetSampler
    \/ \E demand \in Demand : CompleteWindow(demand)
    \/ Send \/ FirstApply \/ ExactRetry \/ HttpReady \/ IndependentSwitch
    \/ \E mode \in Modes : Expire(mode)
    \/ \E group \in Catalog : ReportFailure(group)

Spec == Init /\ [][Next]_vars

EvidenceIsStable ==
    /\ Len(local.support) \in 0..2
    /\ \A i \in DOMAIN local.support :
        LET basis == local.support[i]
        IN /\ basis.revision = local.revision
           /\ basis.incumbent \in Modes /\ basis.target \in Modes /\ basis.incumbent # basis.target
           /\ basis.target = Target(basis.demand)
           /\ basis.demand \notin {"NoGain", "LowerServed"}
           /\ basis.incumbent = local.support[1].incumbent /\ basis.target = local.support[1].target

FreshIntervalsOnly ==
    /\ ~ENABLED (Seed /\ local'.support # local.support)
    /\ ~ENABLED (ShortPoll /\ local'.support # local.support)
    /\ ~ENABLED (ReloadPolicy /\ (local'.support # <<>> \/ local'.age # "Unseeded"))
    /\ ~ENABLED (ResetSampler /\ (local'.support # <<>> \/ local'.age # "Unseeded"))
    /\ \A demand \in Demand :
        ~ENABLED (CompleteWindow(demand) /\
            (local.age # "Full" \/ ~local.read \/ local'.age # "Short" \/ local'.read))

ObservationIsOneCut ==
    ~ENABLED (Observe /\ local'.observed # native.active)

WindowReadiness ==
    \A demand \in Demand :
        ~ENABLED (CompleteWindow(demand) /\ Len(local'.support) > 0 /\
            (local.observed \notin Modes \cap profiles \/ Target(demand) \notin profiles \/
             \E group \in Target(demand) :
                group \notin native.http \/
                IF group \in local.observed THEN group \notin native.active ELSE group \notin native.ready))

StableSubmission ==
    request # NoIntent =>
        /\ Len(local.support) = 2
        /\ request.expected = local.support[2].incumbent /\ request.target = local.support[2].target

ProfilesValidAtSend ==
    ~ENABLED (Send /\ ~({request'.expected, request'.target} \subseteq profiles))

ApplyComparesAllGroups ==
    (request # NoIntent /\ receipt = Pending) =>
        LET canApply == \A group \in Catalog :
                 /\ (group \in request.expected) = (group \in native.active)
                 /\ group \in request.target => group \in native.active \cup native.ready
        IN /\ ENABLED FirstApply
           /\ IF canApply THEN ~ENABLED (FirstApply /\ (receipt' # Applied \/ native'.active # request.target))
              ELSE ~ENABLED (FirstApply /\ (receipt' # Rejected \/ native' # native))

RetryPreservesReceipt ==
    /\ request # NoIntent /\ receipt # Pending /\ ~repeated => ENABLED ExactRetry
    /\ ~ENABLED (ExactRetry /\ (request' # request \/ receipt' # receipt \/ native' # native))
====
