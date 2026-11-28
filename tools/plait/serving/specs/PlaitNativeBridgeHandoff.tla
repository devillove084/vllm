---- MODULE PlaitNativeBridgeHandoff ----
EXTENDS Naturals

CONSTANTS Requests, CheckRelease, CheckRejection, CheckReplacement
Epochs == 1..3
VARIABLES active, observed, ready, held, closed, requests
vars == <<active, observed, ready, held, closed, requests>>

Init ==
    /\ active = 1 /\ observed = 1
    /\ ready = {1, 2} /\ held = {1, 2} /\ closed = {}
    /\ requests = [r \in Requests |->
        [phase |-> "New", route |-> 0, admissions |-> 0,
         admitted |-> 0, output |-> 0, retried |-> FALSE,
         safe |-> TRUE, different |-> TRUE]]

\* Epoch 2 has a separate lease. Epochs 1 and 3 need the same devices.
Switch ==
    /\ active + 1 \in ready \cap held
    /\ active' = active + 1
    /\ UNCHANGED <<observed, ready, held, closed, requests>>

Observe ==
    /\ observed # active
    /\ observed' = active
    /\ UNCHANGED <<active, ready, held, closed, requests>>

Close(epoch) ==
    /\ epoch < active /\ epoch \in held \ closed
    /\ closed' = closed \cup {epoch}
    /\ UNCHANGED <<active, observed, ready, held, requests>>

Release(epoch) ==
    /\ epoch \in held \cap closed
    /\ \A r \in Requests :
        requests[r].phase \notin {"Streaming", "Ambiguous"} \/ requests[r].admitted # epoch
    /\ held' = held \ {epoch}
    /\ UNCHANGED <<active, observed, ready, closed, requests>>

PrepareTarget ==
    /\ active = 2 /\ 3 \notin held
    /\ (~CheckRelease \/ 1 \notin held)
    /\ held' = held \cup {3}
    /\ UNCHANGED <<active, observed, ready, closed, requests>>

ReadyTarget ==
    /\ 3 \in held \ ready
    /\ ready' = ready \cup {3}
    /\ UNCHANGED <<active, observed, held, closed, requests>>

Send(r) ==
    /\ requests[r].phase = "New"
    /\ requests' = [requests EXCEPT ![r].phase = "Sent", ![r].route = observed]
    /\ UNCHANGED <<active, observed, ready, held, closed>>

Reject(r) ==
    /\ requests[r].phase = "Sent" /\ requests[r].route \in closed
    /\ requests' = [requests EXCEPT ![r].phase = "Rejected"]
    /\ UNCHANGED <<active, observed, ready, held, closed>>

Admit(r) ==
    /\ requests[r].phase = "Sent" /\ requests[r].route \in (held \cap ready) \ closed
    /\ requests' = [requests EXCEPT ![r].phase = "Streaming",
         ![r].admitted = requests[r].route, ![r].admissions = @ + 1]
    /\ UNCHANGED <<active, observed, ready, held, closed>>

LoseReply(r) ==
    /\ requests[r].phase = "Streaming"
    /\ requests' = [requests EXCEPT ![r].phase = "Ambiguous"]
    /\ UNCHANGED <<active, observed, ready, held, closed>>

Retry(r) ==
    /\ ~requests[r].retried
    /\ requests[r].phase \in {"Rejected", "Ambiguous"}
    /\ (~CheckRejection \/ requests[r].phase = "Rejected")
    /\ (~CheckReplacement \/ requests[r].route # observed)
    /\ requests' = [requests EXCEPT ![r].phase = "Sent", ![r].route = observed,
         ![r].retried = TRUE, ![r].safe = requests[r].phase = "Rejected",
         ![r].different = requests[r].route # observed]
    /\ UNCHANGED <<active, observed, ready, held, closed>>

Expire(r) ==
    /\ requests[r].phase \in {"Rejected", "Ambiguous"}
    /\ requests' = [requests EXCEPT ![r].phase = "Failed"]
    /\ UNCHANGED <<active, observed, ready, held, closed>>

Finish(r) ==
    /\ requests[r].phase = "Streaming"
    /\ requests' = [requests EXCEPT ![r].phase = "Done", ![r].output = requests[r].admitted]
    /\ UNCHANGED <<active, observed, ready, held, closed>>

Next ==
    \/ Switch \/ Observe \/ PrepareTarget \/ ReadyTarget
    \/ \E e \in Epochs : Close(e) \/ Release(e)
    \/ \E r \in Requests :
        Send(r) \/ Reject(r) \/ Admit(r) \/ LoseReply(r) \/ Retry(r) \/ Expire(r) \/ Finish(r)

Spec == Init /\ [][Next]_vars
ActiveIsReady == active \in (ready \cap held) \ closed
ExclusiveSharedDevices == ~({1, 3} \subseteq held)
SingleAdmission == \A r \in Requests : requests[r].admissions <= 1
VerifiedRejection == \A r \in Requests : requests[r].safe
DifferentRoute == \A r \in Requests : requests[r].different
PinnedOutput == \A r \in Requests :
    requests[r].output = 0 \/ requests[r].output = requests[r].admitted
====
