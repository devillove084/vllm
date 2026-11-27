---- MODULE PlaitCheckpointBudgetSwitch ----
EXTENDS Integers, FiniteSets

CONSTANTS Stores, MaxGeneration
Plans == {"initial", "x", "y"}
None == "none"
VARIABLES generation, active, previous, target, pending, attempt, seen
vars == <<generation, active, previous, target, pending, attempt, seen>>

Init ==
    /\ generation = [s \in Stores |-> 0]
    /\ active = [s \in Stores |-> "initial"]
    /\ previous = [s \in Stores |-> None]
    /\ target = [s \in Stores |-> None]
    /\ pending = [s \in Stores |-> 0]
    /\ attempt = [s \in Stores |-> None]
    /\ seen = [s \in Stores |-> {}]

Grant(s) ==
    /\ target[s] = None
    /\ pending[s] = 0
    /\ pending' = [pending EXCEPT ![s] = 1]
    /\ UNCHANGED <<generation, active, previous, target, attempt, seen>>

Drain(s) ==
    /\ pending[s] > 0
    /\ pending' = [pending EXCEPT ![s] = 0]
    /\ UNCHANGED <<generation, active, previous, target, attempt, seen>>

Prepare(s, plan) ==
    /\ attempt[s] = None
    /\ generation[s] < MaxGeneration
    /\ plan # active[s]
    /\ target' = [target EXCEPT ![s] = plan]
    /\ UNCHANGED <<generation, active, previous, pending, attempt, seen>>

Begin(s) ==
    /\ attempt[s] = None
    /\ target[s] # None
    /\ pending[s] = 0
    /\ attempt' = [attempt EXCEPT ![s] = target[s]]
    /\ UNCHANGED <<generation, active, previous, target, pending, seen>>

Observe(s, peer) ==
    /\ attempt[s] # None
    /\ peer # s
    /\ peer \notin seen[s]
    /\ \/ /\ generation[peer] = generation[s]
          /\ active[peer] = active[s]
          /\ target[peer] = attempt[s]
          /\ pending[peer] = 0
       \/ /\ generation[peer] = generation[s] + 1
          /\ active[peer] = attempt[s]
          /\ previous[peer] = active[s]
    /\ seen' = [seen EXCEPT ![s] = @ \cup {peer}]
    /\ UNCHANGED <<generation, active, previous, target, pending, attempt>>

Activate(s) ==
    /\ attempt[s] # None
    /\ seen[s] = Stores \ {s}
    /\ generation' = [generation EXCEPT ![s] = @ + 1]
    /\ active' = [active EXCEPT ![s] = attempt[s]]
    /\ previous' = [previous EXCEPT ![s] = active[s]]
    /\ target' = [target EXCEPT ![s] = None]
    /\ attempt' = [attempt EXCEPT ![s] = None]
    /\ seen' = [seen EXCEPT ![s] = {}]
    /\ UNCHANGED pending

Retry(s) ==
    /\ attempt[s] # None
    /\ attempt' = [attempt EXCEPT ![s] = None]
    /\ seen' = [seen EXCEPT ![s] = {}]
    /\ UNCHANGED <<generation, active, previous, target, pending>>

\* Restart preserves the durable plan and gate, and loses the grants and
\* observations that belonged to the stopped process.
Restart(s) ==
    /\ pending' = [pending EXCEPT ![s] = 0]
    /\ attempt' = [attempt EXCEPT ![s] = None]
    /\ seen' = [seen EXCEPT ![s] = {}]
    /\ UNCHANGED <<generation, active, previous, target>>

Next ==
    \E s \in Stores :
        \/ Grant(s) \/ Drain(s) \/ Begin(s) \/ Activate(s) \/ Retry(s) \/ Restart(s)
        \/ \E plan \in Plans : Prepare(s, plan)
        \/ \E peer \in Stores : Observe(s, peer)

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ generation \in [Stores -> 0..MaxGeneration]
    /\ active \in [Stores -> Plans]
    /\ previous \in [Stores -> Plans \cup {None}]
    /\ target \in [Stores -> Plans \cup {None}]
    /\ pending \in [Stores -> 0..1]
    /\ attempt \in [Stores -> Plans \cup {None}]
    /\ seen \in [Stores -> SUBSET Stores]

OnePlanPerGeneration ==
    \A s, peer \in Stores :
        generation[s] = generation[peer] => active[s] = active[peer]

RetiredGenerationDrained ==
    \A s, peer \in Stores :
        generation[s] < generation[peer] => target[s] # None /\ pending[s] = 0
====
