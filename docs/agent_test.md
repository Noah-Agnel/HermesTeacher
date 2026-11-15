Test de l'agent — mentions administratives
==========================================

Objectif : vérifier que l'agent n'extrait pas les mentions administratives
(établissement, auteur, date, pagination) comme des concepts.

Contexte
--------
Le `DemoGenerator` historicallement utilisé découpait le PDF ligne par ligne :
la page de garde devenait un concept. Ce comportement a été supprimé. Ce test
vérifie que l'agent, bien guidé, ne fait pas l'erreur.

Procédure
---------
1. Préparer un PDF de cours dont la première page contient une page de garde
   avec le nom de l'établissement, l'auteur et l'année.
2. Créer un concept et déposer ce PDF (via l'interface `/concepts/new`).
3. Demander à l'agent (skill `hermes-teacher`) de proposer le graphe.
4. Lire le graphe enregistré :

       python -m backend.tutor.agent_tools graph <concept_id>

Critère de réussite
-------------------
Le graphe ne contient aucun concept issu de la page de garde. En particulier,
le nom de l'établissement, l'auteur et l'année n'apparaissent pas comme
concepts.

Si un concept de ce type apparaît
--------------------------------
Corriger la section « Attribution des documents » du skill `hermes-teacher`
et recommencer. Le backend ne peut pas juger du contenu : il vérifie seulement
la structure (DAG, documents cités, concepts).
