---- MODULE PlaitAutomaticDeviceReuse ----
EXTENDS Naturals

CONSTANTS RequireForecast, RequireCoverage, RequireRelease, RequireReady
VARIABLES active, eligible, bridgeReady, covers, forecast, oldHeld, newHeld, ready,
          certified, targetCertified, restarts
vars == <<active, eligible, bridgeReady, covers, forecast, oldHeld, newHeld, ready,
          certified, targetCertified, restarts>>

Init ==
    /\ active = "Source"
    /\ eligible \in BOOLEAN /\ bridgeReady \in BOOLEAN /\ covers \in BOOLEAN
    /\ forecast = FALSE /\ oldHeld = TRUE /\ newHeld = FALSE /\ ready = FALSE
    /\ certified = TRUE /\ targetCertified = TRUE /\ restarts = 0

\* Device availability and observations can change while traffic is served.
UpdateInputs(e, b, c) ==
    /\ eligible' = e /\ bridgeReady' = b /\ covers' = c
    /\ UNCHANGED <<active, forecast, oldHeld, newHeld, ready, certified,
                   targetCertified, restarts>>

Preview ==
    /\ active = "Source"
    /\ forecast' = eligible
    /\ UNCHANGED <<active, eligible, bridgeReady, covers, oldHeld, newHeld,
                   ready, certified, targetCertified, restarts>>

ActivateBridge ==
    /\ active = "Source"
    /\ (~RequireForecast \/ forecast)
    /\ (~RequireCoverage \/ (bridgeReady /\ covers))
    /\ active' = "Bridge"
    /\ certified' = (forecast /\ bridgeReady /\ covers)
    /\ UNCHANGED <<eligible, bridgeReady, covers, forecast, oldHeld, newHeld,
                   ready, targetCertified, restarts>>

ReleaseSource ==
    /\ active = "Bridge" /\ oldHeld
    /\ oldHeld' = FALSE
    /\ UNCHANGED <<active, eligible, bridgeReady, covers, forecast, newHeld,
                   ready, certified, targetCertified, restarts>>

PrepareTarget ==
    /\ active = "Bridge" /\ eligible /\ ~newHeld
    /\ (~RequireRelease \/ ~oldHeld)
    /\ newHeld' = TRUE
    /\ UNCHANGED <<active, eligible, bridgeReady, covers, forecast, oldHeld,
                   ready, certified, targetCertified, restarts>>

ReadyTarget ==
    /\ newHeld /\ ~ready
    /\ ready' = TRUE
    /\ UNCHANGED <<active, eligible, bridgeReady, covers, forecast, oldHeld,
                   newHeld, certified, targetCertified, restarts>>

ActivateTarget ==
    /\ active = "Bridge"
    /\ (~RequireReady \/ ready)
    /\ active' = "Target"
    /\ targetCertified' = ready
    /\ UNCHANGED <<eligible, bridgeReady, covers, forecast, oldHeld, newHeld,
                   ready, certified, restarts>>

Restart ==
    /\ restarts = 0
    /\ restarts' = 1
    /\ forecast' = FALSE
    /\ UNCHANGED <<active, eligible, bridgeReady, covers, oldHeld, newHeld,
                   ready, certified, targetCertified>>

Next ==
    \/ \E e, b, c \in BOOLEAN : UpdateInputs(e, b, c)
    \/ Preview \/ ActivateBridge \/ ReleaseSource \/ PrepareTarget
    \/ ReadyTarget \/ ActivateTarget \/ Restart

Spec == Init /\ [][Next]_vars
BridgeAdmission == certified
SingleOwner == ~(oldHeld /\ newHeld)
TargetAdmission == targetCertified
RestartCanContinue ==
    active = "Bridge" /\ restarts = 1 /\ ~oldHeld /\ eligible /\ ~newHeld
    => ENABLED PrepareTarget
====
