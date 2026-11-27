---- MODULE PlaitServingIngress ----
EXTENDS Naturals

CONSTANTS Requests, NativeRoutes, EnforceVersion
VARIABLES active, route, oldOpen, origins, requests

Epochs == 1..2
vars == <<active, route, oldOpen, origins, requests>>
GateOpen(e) == IF e = 1 THEN oldOpen ELSE active = e
CanAdmit(r) ==
    /\ GateOpen(requests[r].upstream)
    /\ (EnforceVersion => requests[r].upstream = requests[r].target)

Init ==
    /\ active = 1 /\ route = 1 /\ oldOpen = TRUE
    /\ origins = [e \in Epochs |-> IF NativeRoutes /\ e = 2 THEN 0 ELSE e]
    /\ requests = [r \in Requests |->
        [phase |-> "Pending", target |-> 0, upstream |-> 0, admissions |-> 0,
         admittedOn |-> 0, outputOn |-> {}, chunks |-> 0]]

Activate ==
    /\ active = 1 /\ active' = 2
    /\ UNCHANGED <<route, oldOpen, origins, requests>>

\* The ingress can continue to see the previous epoch during a refresh.
Observe(epoch) ==
    /\ epoch <= active /\ epoch >= route
    /\ route' = epoch
    /\ UNCHANGED <<active, oldOpen, origins, requests>>

\* The deployment supplies an HTTP origin for each observed native epoch.
Bind(epoch, upstream) ==
    /\ NativeRoutes
    /\ origins' = [origins EXCEPT ![epoch] = upstream]
    /\ UNCHANGED <<active, route, oldOpen, requests>>

CloseOld ==
    /\ active = 2 /\ oldOpen
    /\ oldOpen' = FALSE
    /\ UNCHANGED <<active, route, origins, requests>>

Connect(r) ==
    /\ requests[r].phase = "Pending"
    /\ origins[route] # 0
    /\ requests' = [requests EXCEPT ![r].phase = "Connecting", ![r].target = route,
        ![r].upstream = origins[route]]
    /\ UNCHANGED <<active, route, oldOpen, origins>>

Send(r) ==
    /\ requests[r].phase = "Connecting"
    /\ requests' = [requests EXCEPT ![r].phase = "Sent"]
    /\ UNCHANGED <<active, route, oldOpen, origins>>

Retry(r) ==
    /\ (requests[r].phase = "Connecting"
        \/ (requests[r].phase = "Sent" /\ ~CanAdmit(r)))
    /\ requests' = [requests EXCEPT ![r].phase = "Pending", ![r].target = 0,
        ![r].upstream = 0]
    /\ UNCHANGED <<active, route, oldOpen, origins>>

Admit(r) ==
    /\ requests[r].phase = "Sent" /\ CanAdmit(r)
    /\ requests' = [requests EXCEPT ![r].phase = "Streaming",
        ![r].admittedOn = requests[r].upstream, ![r].admissions = @ + 1]
    /\ UNCHANGED <<active, route, oldOpen, origins>>

AmbiguousFailure(r) ==
    /\ requests[r].phase \in {"Sent", "Streaming"}
    /\ requests' = [requests EXCEPT ![r].phase = "Failed"]
    /\ UNCHANGED <<active, route, oldOpen, origins>>

Forward(r) ==
    /\ requests[r].phase = "Streaming"
    /\ requests' = [requests EXCEPT ![r].chunks = @ + 1,
        ![r].outputOn = @ \cup {requests[r].upstream},
        ![r].phase = IF requests[r].chunks = 1 THEN "Done" ELSE "Streaming"]
    /\ UNCHANGED <<active, route, oldOpen, origins>>

Next == Activate \/ CloseOld \/ (\E e \in Epochs : Observe(e))
    \/ (\E e \in Epochs, upstream \in 0..2 : Bind(e, upstream))
    \/ (\E r \in Requests :
        Connect(r) \/ Send(r) \/ Retry(r) \/ Admit(r) \/ AmbiguousFailure(r) \/ Forward(r))

Spec == Init /\ [][Next]_vars

SingleAdmission == \A r \in Requests : requests[r].admissions <= 1
VersionBound == \A r \in Requests :
    requests[r].admittedOn # 0 => requests[r].admittedOn = requests[r].target
PinnedOutput == \A r \in Requests : requests[r].outputOn \subseteq {requests[r].admittedOn}
NoRouteRegression == [][route' >= route]_vars
OldStreamCanFinish == \A r \in Requests :
    active = 2 /\ ~oldOpen /\ requests[r].phase = "Streaming" /\ requests[r].admittedOn = 1
    => ENABLED Forward(r)
====
