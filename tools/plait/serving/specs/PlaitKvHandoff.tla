---- MODULE PlaitKvHandoff ----
EXTENDS FiniteSets

\* A context identifies the model namespace, prompt and cache salt.
\* A receipt can survive a reset of the shared page store.
Contexts == {"a", "b"}
Required == {1, 2}

VARIABLES pages, receipts, requested, offered, loaded, phase
vars == <<pages, receipts, requested, offered, loaded, phase>>

Init ==
    /\ pages = [context \in Contexts |-> {}]
    /\ receipts = {}
    /\ requested = "none" /\ offered = "none"
    /\ loaded = {}
    /\ phase = "waiting"

PublishPage(context, page) ==
    /\ pages' = [pages EXCEPT ![context] = @ \cup {page}]
    /\ UNCHANGED <<receipts, requested, offered, loaded, phase>>

Export(context) ==
    /\ pages[context] = Required
    /\ receipts' = receipts \cup {context}
    /\ UNCHANGED <<pages, requested, offered, loaded, phase>>

ResetStore ==
    /\ pages' = [context \in Contexts |-> {}]
    /\ UNCHANGED <<receipts, requested, offered, loaded, phase>>

Admit(context, receipt) ==
    /\ phase = "waiting" /\ receipt \in receipts
    /\ context = receipt
    /\ pages[context] = Required
    /\ requested' = context /\ offered' = receipt
    /\ phase' = "loading"
    /\ UNCHANGED <<pages, receipts, loaded>>

Fetch(page) ==
    /\ phase = "loading" /\ page \in Required \ loaded
    /\ IF page \in pages[requested]
          THEN /\ loaded' = loaded \cup {page}
               /\ UNCHANGED phase
          ELSE /\ phase' = "failed"
               /\ UNCHANGED loaded
    /\ UNCHANGED <<pages, receipts, requested, offered>>

Emit ==
    /\ phase = "loading" /\ loaded = Required
    /\ phase' = "served"
    /\ UNCHANGED <<pages, receipts, requested, offered, loaded>>

Next ==
    \/ \E c \in Contexts, i \in Required : PublishPage(c, i)
    \/ \E c \in Contexts : Export(c)
    \/ \E c, r \in Contexts : Admit(c, r)
    \/ \E i \in Required : Fetch(i)
    \/ ResetStore \/ Emit

Spec == Init /\ [][Next]_vars

MatchingContext == phase # "waiting" => requested = offered /\ offered \in receipts
CompleteLoadBeforeOutput == phase = "served" => loaded = Required
====
