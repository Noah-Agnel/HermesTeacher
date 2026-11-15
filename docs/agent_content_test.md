Test de l'agent — génération de contenu
=======================================

Objectif
--------
Vérifier que l'agent sait produire, à partir des PDF déposés, une proposition de
graphe, une évaluation initiale, un mini-cours et un exercice, et que le
backend les valide et les persiste jusqu'à l'interface.

Principe
--------
L'agent joue son propre rôle : il lit le profil avec `profile` et les PDF avec
`read_file`, puis propose le contenu. Deux modes :

- **direct** (cette procédure) : l'agent écrit un JSON dans un dossier de
  travail et appelle l'outil `save_*` correspondant ;
- **interface web** : `HermesGenerator` appelle l'agent en sous-processus
  `hermes -z`, l'agent répond directement avec le JSON (voir plus bas).

Dans les deux cas, le backend valide, crée les identifiants, met à jour le
parcours et persiste.

Préparer un concept de test
--------------------------
Utiliser un dossier de données isolé pour ne pas toucher à `data/` :

    TUTOR_DATA_DIR=/tmp/hermesteacher-manual-test

Créer le concept et déposer les documents (le ploy et les TD) avec une petit
script qui enchaîne `ProfileStore.create_concept`, `SQLiteStore.create_learner`
et `documents.store_upload`, puis `store.save_document`. Noter le
`concept_id` et les `document_id` renvoyés.

Procédure
---------
1. Lire l'état :

       python -m backend.tutor.agent_tools concepts
       python -m backend.tutor.agent_tools documents <concept_id>

2. Lire le contenu des PDF (chemins renvoyés par `documents`) avec `read_file`.

3. Proposer le graphe : écrire `graph.json` puis

       python -m backend.tutor.agent_tools save_graph <concept_id> /tmp/agent-work/graph.json

   La phase reste `not_started`.

4. Proposer l'évaluation initiale : écrire `evaluation.json` puis

       python -m backend.tutor.agent_tools save_evaluation <concept_id> /tmp/agent-work/evaluation.json

   La phase passe à `initial_evaluation`.

5. Soumettre l'évaluation pour atteindre la phase `learning`. L'agent lit
   l'activité (`activity`), écrit les réponses et ses notes dans
   `data/agent/grade.json`, puis :

       python -m backend.tutor.agent_tools submit_response <concept_id> /tmp/agent-work/grade.json

   La phase passe à `learning`.

6. Proposer le cours et l'exercice sur le concept que le backend attend :

       python -m backend.tutor.agent_tools mastery <concept_id>
       python -m backend.tutor.agent_tools save_course <concept_id> /tmp/agent-work/course.json
       python -m backend.tutor.agent_tools save_exercise <concept_id> /tmp/agent-work/exercise.json

   Si le concept visé n'est pas le bon, l'erreur renvoie le concept attendu.

7. Vérifier dans l'interface : la page du concept affiche l'exercice généré,
   son barème et ses citations, et le graphe propose les concepts du ploy.

Résultats observés
------------------
Pour le concept « Chaînes de Markov » (ploy + TD1, TD2, TD3) :

- graphe : 9 concepts, 8 arêtes, aucune mention administrative ;
- évaluation initiale : 7 questions, une par notion, avec citations page +
  extrait ;
- cours : « Variables aléatoires et espérance », 5 sections, 4 citations ;
- exercice : loi exponentielle (inversion + espérance conditionnelle), 3
  citations ;
- le garde-fou d'ordre refuse un cours sur un concept qui n'est pas le suivant.

Limites
-------
- L'examen final n'a été testé que par les tests unitaires ; il utilise le même
  chemin que l'évaluation initiale.
- Le backend ne vérifie pas que le contenu est pédagogiquement correct : il
  vérifie la structure, les coefficients, les concepts et les citations.

Tester le prototype complet (interface web + agent)
--------------------------------------------------
Prérequis : `hermes` installé et configuré sur la machine, dépôt cloné,
dépendances installées (`pip install -r requirements.txt`).

1. Isoler les données (optionnel) :

       export TUTOR_DATA_DIR=/tmp/hermesteacher-web-test

2. Lancer l'interface depuis la racine du dépôt :

       python -m backend.web

   Le sous-processus agent démarre dans le dépôt pour trouver le skill de
   projet et les commandes `agent_tools`.

3. Ouvrir http://127.0.0.1:8000 et créer un concept : titre, objectifs,
   préférences (format, focus, difficultés, temps), et déposer un ou plusieurs
   PDF.

4. Valider : l'interface affiche l'évaluation initiale générée par l'agent.

5. Répondre à l'évaluation, puis utiliser les boutons « Next lesson » et
   « Next exercice » pour générer le cours et l'exercice.

6. Vérifier le profil global : `/profile` affiche les préférences et les
   points faibles. L'agent les lit avec `profile` et peut les mettre à jour
   avec `save_profile`.

7. Vérifier les niveaux : la barre de progression et le graphe se mettent à
   jour après chaque exercice.

Pour déboguer l'agent : lancer la même commande à la main depuis le dépôt,

    hermes -z "Génère un exercice pour le concept X"

ou consulter les sessions avec `hermes sessions browse`.

Tests automatisés
-----------------
    pip install -r requirements-dev.txt
    python -m unittest discover

Les tests utilisent le `ScriptedGenerator` (double déterministe) pour exercer
l'interface et le moteur sans agent ; les tests de `HermesGenerator` remplacent
l'agent par une commande factice qui renvoie un JSON fixe.
