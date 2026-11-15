# Outils du tuteur — détail

## Convention

- Exécuter depuis la racine du dépôt (`/workspace`).
- Une seule commande par appel `terminal`.
- Parser le JSON de sortie ; ne jamais afficher le JSON brut à l'élève.

## Commandes

### Lister les concepts

    python -m backend.tutor.agent_tools concepts

Retourne la liste des concepts avec leur phase.

### Lire les documents d'un concept

    python -m backend.tutor.agent_tools documents <concept_id>

Retourne les documents déposés (document_id, title, path, media_type).

### Lire le graphe de prérequis

    python -m backend.tutor.agent_tools graph <concept_id>

Retourne la phase et le graphe (concepts + arêtes).

### Lire la maîtrise

    python -m backend.tutor.agent_tools mastery <concept_id>

Retourne le niveau actuel, le niveau cible et le poids de chaque concept.

### Lire l'historique

    python -m backend.tutor.agent_tools history <concept_id>

Retourne les événements (document déposé, cours lu, exercice soumis, ...).

### Lire les erreurs

    python -m backend.tutor.agent_tools errors <concept_id>

Retourne les erreurs commises, avec leur état de résolution.

### Lire l'activité en cours

    python -m backend.tutor.agent_tools activity <concept_id>

Retourne la phase, le concept en cours et les questions de l'assessment en
cours (question_id, prompt, réponse attendue, barème, coefficient de
difficulté). C'est ce qu'il faut pour corriger.

### Lire le profil global

    python -m backend.tutor.agent_tools profile

Retourne le profil de l'élève (préférences, points faibles transverses).

### Écrire le profil global

    python -m backend.tutor.agent_tools save_profile <chemin>

Le profil est la source de vérité des préférences et des points faibles
transverses de l'élève. L'agent écrit un fichier JSON dans `data/agent/` :

    {
      "display_name": "Prénom Nom",
      "preferences": {"format": "exemples"},
      "weaknesses": ["démonstrations"]
    }

Les trois champs sont optionnels ; les champs absents gardent leur valeur
actuelle. Le backend valide et persiste. Le profil n'est pas injecté dans les
prompts : l'agent le lit avec la commande `profile` quand il en a besoin.

## Format de sortie

    {"ok": true, "data": [...]}
    {"ok": false, "error": "concept inconnu : 'nope'"}

Le code de sortie vaut 0 en cas de succès, 1 en cas d'erreur.

## Lecture du contenu des documents

Les documents sont des PDF stockés sur disque. À l'import, le backend crée à
côté de chaque PDF un fichier texte (même chemin, extension `.txt`) avec un
marqueur `=== Page N ===` par page et les formules converties en Unicode
(∑, ∫, ℝ, indices `X_{n}`). Lis ce `.txt` avec `read_file` plutôt que le PDF.
Quand le prompt contient déjà des balises `<document>`, le contenu est fourni :
ne relis pas les fichiers. Ne demande pas à l'élève de copier le contenu.

## Proposer un graphe

### Écrire la proposition

L'agent écrit un fichier JSON dans `data/agent/` :

    {
      "graph_id": "string",
      "source_document_ids": ["string"],
      "notes": "string",
      "concepts": [
        {
          "concept_id": "string",
          "label": "string",
          "description": "string",
          "target_level": 1.0,
          "weight": 1.0,
          "evidence": [{"document_id": "string", "page": 1, "quote": "string"}]
        }
      ],
      "edges": [
        {"prerequisite_id": "string", "concept_id": "string", "evidence": []}
      ]
    }

### Enregistrer la proposition

    python -m backend.tutor.agent_tools save_graph <concept_id> <chemin>

Le backend valide : le graphe doit être un DAG, les concepts doivent exister,
les documents cités doivent exister. En cas d'échec, la réponse contient la liste
des problèmes ; corrige le fichier et relance la commande.

Le dossier `data/agent/` n'est pas créé par le backend : crée-le avant d'écrire
la proposition. Le champ `error` est une chaîne qui emballe un JSON (liste
d'issues avec `code` et `message`) : parse-le pour corriger, ne l'affiche pas
brut à l'élève.

Réponse de succès :

    {"ok": true, "data": {"graph_id": "...", "concepts": 8, "edges": 7}}

## Proposer du contenu

Quatre commandes, même principe : écrire le JSON dans `data/agent/` et passer
le chemin. Le backend valide, crée les identifiants et met à jour le parcours.

### save_evaluation <concept_id> <chemin>

Phase requise : `not_started`. Enregistre l'évaluation initiale et passe la
phase à `initial_evaluation`.

### save_course <concept_id> <chemin>

Phase requise : `learning`. Enregistre le mini-cours et note le concept en
cours.

### save_exercise <concept_id> <chemin>

Phase requise : `learning`. Crée l'assessment `exercise` et la question, puis
le note en cours.

### save_final_exam <concept_id> <chemin>

Phase requise : `learning`. Enregistre l'examen final et passe la phase à
`final_exam`.

### submit_response <concept_id> <chemin>

Phase requise : une évaluation ou un exercice en cours. Enregistre les notes
que l'agent propose, puis le backend calcule les niveaux.

Schéma :

    {
      "responses": [
        {
          "question_id": "string",
          "response_text": "string",
          "score": 0.8,
          "assistance_coefficient": 1.0,
          "feedback": "string"
        }
      ]
    }

Chaque réponse soumise doit porter une note, et chaque note doit viser une
question de l'assessment en cours. Réponse de succès :

    {"ok": true, "data": {"score": 0.8, "assistance_coefficient": 1.0,
                          "feedback": "...", "level": 0.62, "gap": 1.0,
                          "progress": 0.0}}

### Schémas attendus

Cours :

    {
      "concept_id": "string",
      "title": "string",
      "content": "string",
      "citations": [{"document_id": "string", "page": 1, "quote": "string"}]
    }

Exercice :

    {
      "question_id": "string",
      "concept_id": "string",
      "prompt": "string",
      "reference_answer": "string",
      "grading_rubric": "string",
      "difficulty_coefficient": 1.0,
      "citations": [{"document_id": "string", "page": 1, "quote": "string"}]
    }

Évaluation initiale et examen final :

    {"questions": [{question ci-dessus}, ...]}

### Réponses de succès

    {"ok": true, "data": {"course_id": "...", "concept_id": "...", "title": "..."}}
    {"ok": true, "data": {"assessment_id": "...", "question_id": "...", "concept_id": "..."}}
    {"ok": true, "data": {"assessment_id": "...", "questions": 7, "question_ids": ["..."]}}

### Erreurs fréquentes

- `concept 'x' pas prévu maintenant ; le backend attend : y` — vise le concept
  renvoyé, pas un autre.
- `phase attendue 'learning', parcours en 'initial_evaluation'` — l'évaluation
  n'a pas encore été soumise.
- `concept 'x' hors du graphe` / `unknown document 'x'` — le JSON vise un
  concept ou un document qui n'existe pas.
- `missing grades for questions: [...]` — une réponse soumise n'est pas notée.
- `grades reference unknown questions: [...]` — la note vise une question hors
  de l'assessment en cours.
- `question attendue : 'x'` — l'exercice en cours porte une autre question.

## Ce qu'il ne faut pas faire

- Ne pas inventer de concept sans extrait.
- Ne pas prendre les mentions administratives (établissement, auteur, date) pour des concepts.
- Ne pas renvoyer un niveau, un écart ou un ordre : le backend les calcule.
- Ne pas citer un document qui n'existe pas : le backend rejette la proposition.
