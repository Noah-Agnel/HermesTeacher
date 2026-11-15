"""FastAPI interface for the tutor.

The interface is intentionally simple: a welcome page listing the concepts in
progress, a concept page with the progress bar and the concept graph, a page
to deposit documents and start a new concept, and a settings page. Every
learning-path decision is delegated to the backend engine.
"""

from __future__ import annotations

import time
from pathlib import Path
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote_plus

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import Settings, load_settings
from .jobs import JobManager
from .service import EvaluationPending, TutorService


PHASE_LABELS = {
    "not_started": "Non démarré",
    "initial_evaluation": "Évaluation initiale",
    "learning": "Apprentissage",
    "final_exam": "Examen final",
    "completed": "Terminé",
}


_FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 26 26">'
    '<rect width="26" height="26" rx="6" fill="#070c1c"/>'
    '<line x1="5" y1="19" x2="12" y2="8" stroke="#5b9dff" stroke-width="1.4"/>'
    '<line x1="12" y1="8" x2="21" y2="13" stroke="#5b9dff" stroke-width="1.4"/>'
    '<circle cx="5" cy="19" r="2.8" fill="#4f84ee"/><circle cx="12" cy="8" r="2.8" fill="#6cc3ff"/>'
    '<circle cx="21" cy="13" r="3.4" fill="#a4f6ff"/></svg>'
)


def phase_label(phase: str) -> str:
    return PHASE_LABELS.get(phase, phase)


def _redirect(concept_id: str, error: str | None = None) -> RedirectResponse:
    url = f"/concepts/{concept_id}"
    if error:
        url += f"?error={quote_plus(error)}"
    return RedirectResponse(url, status_code=303)


def _form_responses(form: Mapping[str, Any], questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the responses of a multi-question form, with "je ne sais pas"."""
    return [
        {
            "question_id": question["question_id"],
            "response_text": form.get(f"response-{question['question_id']}", "") or "",
            "dont_know": bool(form.get(f"dontknow-{question['question_id']}")),
        }
        for question in questions
    ]


def _preferences(
    format_preference: str,
    focus: str,
    difficulties: str,
) -> dict[str, Any]:
    preferences: dict[str, Any] = {"format": format_preference}
    if focus.strip():
        preferences["focus"] = focus.strip()
    if difficulties.strip():
        preferences["difficulties"] = difficulties.strip()
    return preferences


def create_app(settings: Settings | None = None, generator: Any = None) -> FastAPI:
    settings = settings or load_settings()
    service = TutorService(settings, generator=generator)

    app = FastAPI(title="HermesTeacher")
    app.state.service = service
    jobs = JobManager()
    app.state.jobs = jobs
    app.state.settings = settings

    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    templates.env.globals["phase_label"] = phase_label
    # Changes at every server start, so browsers never keep an old CSS/JS.
    templates.env.globals["asset_version"] = str(int(time.time()))
    app.mount(
        "/static",
        StaticFiles(directory=str(Path(__file__).parent / "static")),
        name="static",
    )

    # ------------------------------------------------------------------
    # welcome page
    # ------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def welcome(request: Request):
        concepts = service.list_concepts()
        return templates.TemplateResponse(
            request, "welcome.html", {"concepts": concepts}
        )

    # ------------------------------------------------------------------
    # global learner profile
    # ------------------------------------------------------------------
    @app.get("/profile", response_class=HTMLResponse)
    def profile_page(request: Request):
        return templates.TemplateResponse(
            request, "profile.html", {"profile": service.get_profile()}
        )

    @app.post("/profile")
    async def profile_update(request: Request):
        try:
            form = await request.form()
            preferences = _preferences(
                form.get("format", "texte"),
                form.get("focus", ""),
                form.get("difficulties", ""),
            )
            weaknesses = [
                item.strip()
                for item in (form.get("weaknesses") or "").split("\n")
                if item.strip()
            ]
            service.save_profile(
                display_name=form.get("display_name", "").strip() or None,
                preferences=preferences,
                weaknesses=weaknesses,
            )
        except Exception:
            pass
        return RedirectResponse("/profile", status_code=303)

    # ------------------------------------------------------------------
    # new concept
    # ------------------------------------------------------------------
    @app.get("/concepts/new", response_class=HTMLResponse)
    def concept_new(request: Request):
        return templates.TemplateResponse(
            request,
            "concept_new.html",
            {
                "errors": [],
                "title": "",
                "objectives": "",
                "format": "texte",
                "focus": "",
                "difficulties": "",
                "eta": "0.3",
                "theta": "0.75",
            },
        )

    @app.post("/concepts/new")
    async def concept_create(
        request: Request,
        title: str = Form(...),
        objectives: str = Form(""),
        format_preference: str = Form("texte", alias="format"),
        focus: str = Form(""),
        difficulties: str = Form(""),
        eta: str = Form("0.3"),
        theta: str = Form("0.75"),
        files: list[UploadFile] = File(default=[]),
    ):
        try:
            eta_value = float(eta)
            theta_value = float(theta)
            preferences = _preferences(
                format_preference, focus, difficulties
            )
            uploads = [
                (file.filename or "document", await file.read())
                for file in files
                if file.filename
            ]
            concept_id = service.create_concept(
                title=title.strip(),
                objectives=objectives.strip(),
                preferences=preferences,
                eta=eta_value,
                theta=theta_value,
                uploads=uploads,
            )
        except EvaluationPending as pending:
            return _redirect(
                pending.concept_id,
                "Le graphe est enregistré, mais l'évaluation initiale n'a pas "
                f"pu être générée : {pending}",
            )
        except Exception as error:
            return templates.TemplateResponse(
                request,
                "concept_new.html",
                {
                    "errors": [str(error)],
                    "title": title,
                    "objectives": objectives,
                    "format": format_preference,
                    "focus": focus,
                    "difficulties": difficulties,
                    "eta": eta,
                    "theta": theta,
                },
                status_code=400,
            )
        return RedirectResponse(f"/concepts/{concept_id}", status_code=303)

    # ------------------------------------------------------------------
    # concept detail
    # ------------------------------------------------------------------
    @app.get("/concepts/{concept_id}", response_class=HTMLResponse)
    def concept_detail(concept_id: str, request: Request, error: str | None = None):
        return render_detail(request, concept_id, error)

    def render_detail(
        request: Request,
        concept_id: str,
        error: str | None = None,
        drafts: Mapping[str, Any] | None = None,
        status_code: int = 200,
    ):
        """Render the concept page; ``drafts`` refills the learner's answers."""
        concept = service.get_concept(concept_id)
        if concept is None:
            raise HTTPException(status_code=404, detail="concept not found")
        activity = service.get_activity(concept_id)
        remaining_labels: list[str] = []
        if concept["graph"]:
            labels = {
                item["concept_id"]: item["label"]
                for item in concept["graph"]["concepts"]
            }
            remaining_labels = [
                labels.get(concept_id_, concept_id_) for concept_id_ in concept["remaining"]
            ]
        return templates.TemplateResponse(
            request,
            "concept_detail.html",
            {
                "concept": concept,
                "activity": activity,
                "remaining_labels": remaining_labels,
                "error": error,
                "drafts": dict(drafts or {}),
            },
            status_code=status_code,
        )

    # ------------------------------------------------------------------
    # initial evaluation
    # ------------------------------------------------------------------
    @app.post("/concepts/{concept_id}/evaluation/generate")
    def generate_evaluation(concept_id: str):
        try:
            service.prepare_initial_evaluation(concept_id)
        except Exception as error:
            return _redirect(concept_id, str(error))
        return _redirect(concept_id)

    @app.post("/concepts/{concept_id}/evaluation")
    async def submit_evaluation(concept_id: str, request: Request):
        form = await request.form()
        try:
            activity = service.get_activity(concept_id)
            responses = _form_responses(form, activity.get("questions", []))
            service.submit_initial_evaluation(concept_id, responses)
        except Exception as error:
            # re-render with the answers so the learner does not retype them
            return render_detail(request, concept_id, str(error), form, 400)
        return _redirect(concept_id)

    # ------------------------------------------------------------------
    # learning activities
    # ------------------------------------------------------------------
    @app.post("/concepts/{concept_id}/course/save")
    async def save_course(concept_id: str, request: Request):
        try:
            form = await request.form()
            service.keep_course(concept_id, form.get("course_id", ""))
        except Exception as error:
            return _redirect(concept_id, str(error))
        return _redirect(concept_id)

    @app.post("/concepts/{concept_id}/lesson")
    def next_lesson(concept_id: str):
        try:
            service.next_lesson(concept_id)
        except Exception as error:
            return _redirect(concept_id, str(error))
        return _redirect(concept_id)

    @app.post("/concepts/{concept_id}/exercise")
    def next_exercise(concept_id: str):
        try:
            service.next_exercise(concept_id)
        except Exception as error:
            return _redirect(concept_id, str(error))
        return _redirect(concept_id)

    @app.post("/concepts/{concept_id}/exercise/submit")
    async def submit_exercise(concept_id: str, request: Request):
        form = await request.form()
        try:
            # "je ne sais pas" is sent as an empty answer: graded 0, no model call
            text = "" if form.get("dont_know") else form.get("response_text", "")
            service.submit_exercise(concept_id, text)
        except Exception as error:
            return render_detail(request, concept_id, str(error), form, 400)
        return _redirect(concept_id)

    # ------------------------------------------------------------------
    # final exam and remediation
    # ------------------------------------------------------------------
    @app.post("/concepts/{concept_id}/final-exam")
    def start_final_exam(concept_id: str):
        try:
            service.start_final_exam(concept_id)
        except Exception as error:
            return _redirect(concept_id, str(error))
        return _redirect(concept_id)

    @app.post("/concepts/{concept_id}/final-exam/submit")
    async def submit_final_exam(concept_id: str, request: Request):
        form = await request.form()
        try:
            activity = service.get_activity(concept_id)
            responses = _form_responses(form, activity.get("questions", []))
            service.submit_final_exam(concept_id, responses)
        except Exception as error:
            return render_detail(request, concept_id, str(error), form, 400)
        return _redirect(concept_id)

    @app.post("/concepts/{concept_id}/resume")
    def resume_learning(concept_id: str):
        try:
            service.resume_learning(concept_id)
        except Exception as error:
            return _redirect(concept_id, str(error))
        return _redirect(concept_id)

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    @app.get("/concepts/{concept_id}/settings", response_class=HTMLResponse)
    def concept_settings(concept_id: str, request: Request):
        concept = service.get_concept(concept_id)
        if concept is None:
            raise HTTPException(status_code=404, detail="concept not found")
        return templates.TemplateResponse(
            request,
            "concept_settings.html",
            {"concept": concept, "errors": []},
        )

    @app.post("/concepts/{concept_id}/settings")
    async def concept_settings_update(concept_id: str, request: Request):
        try:
            form = await request.form()
            preferences = _preferences(
                form.get("format", "texte"),
                form.get("focus", ""),
                form.get("difficulties", ""),
            )
            service.update_settings(
                concept_id,
                title=form.get("title", "").strip(),
                objectives=form.get("objectives", "").strip(),
                preferences=preferences,
                eta=float(form.get("eta", "0.3")),
                theta=float(form.get("theta", "0.75")),
            )
        except Exception as error:
            return templates.TemplateResponse(
                request,
                "concept_settings.html",
                {
                    "concept": service.get_concept(concept_id),
                    "errors": [str(error)],
                },
                status_code=400,
            )
        return _redirect(concept_id)

    @app.post("/concepts/{concept_id}/delete")
    def delete_concept(concept_id: str):
        try:
            service.delete_concept(concept_id)
        except Exception:
            pass
        return RedirectResponse("/", status_code=303)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(_FAVICON, media_type="image/svg+xml")

    # ------------------------------------------------------------------
    # graph page
    # ------------------------------------------------------------------
    @app.get("/concepts/{concept_id}/graph", response_class=HTMLResponse)
    def concept_graph(concept_id: str, request: Request):
        concept = service.get_concept(concept_id)
        if concept is None:
            raise HTTPException(status_code=404, detail="concept not found")
        return templates.TemplateResponse(
            request,
            "graph.html",
            {"concept": concept, "graph": service.graph_view(concept_id)},
        )

    # ------------------------------------------------------------------
    # background jobs (live progress); the HTML routes above remain the
    # fallback without JavaScript
    # ------------------------------------------------------------------
    def _start(kind, steps, work, concept_id=None):
        try:
            job = jobs.start(kind, steps, work, concept_id=concept_id)
        except RuntimeError as error:
            return JSONResponse({"error": str(error)}, status_code=409)
        return JSONResponse(job.to_dict(), status_code=202)

    def _known(concept_id: str) -> None:
        if service.profile.get_concept(concept_id) is None:
            raise HTTPException(status_code=404, detail="concept not found")

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job.to_dict()

    @app.post("/api/concepts")
    async def api_create_concept(
        title: str = Form(...),
        objectives: str = Form(""),
        format_preference: str = Form("texte", alias="format"),
        focus: str = Form(""),
        difficulties: str = Form(""),
        eta: str = Form("0.3"),
        theta: str = Form("0.75"),
        files: list[UploadFile] = File(default=[]),
    ):
        try:
            eta_value = float(eta)
            theta_value = float(theta)
            preferences = _preferences(format_preference, focus, difficulties)
        except Exception as error:
            return JSONResponse({"error": str(error)}, status_code=400)
        uploads = [
            (file.filename or "document", await file.read())
            for file in files
            if file.filename
        ]

        def work(job):
            try:
                concept_id = service.create_concept(
                    title=title.strip(),
                    objectives=objectives.strip(),
                    preferences=preferences,
                    eta=eta_value,
                    theta=theta_value,
                    uploads=uploads,
                )
            except EvaluationPending as pending:
                job.fail(
                    "Le graphe est enregistré, mais l'évaluation initiale n'a pas "
                    f"pu être générée : {pending}",
                    redirect=f"/concepts/{pending.concept_id}",
                )
                return
            job.finish(f"/concepts/{concept_id}")

        return _start(
            "create",
            [
                ("documents", "Lecture des documents"),
                ("graph", "Construction du graphe"),
                ("evaluation", "Évaluation initiale"),
            ],
            work,
        )

    @app.post("/api/concepts/{concept_id}/evaluation/generate")
    def api_generate_evaluation(concept_id: str):
        _known(concept_id)

        def work(job):
            service.prepare_initial_evaluation(concept_id)
            job.finish(f"/concepts/{concept_id}")

        return _start("evaluation", [("evaluation", "Évaluation initiale")], work, concept_id)

    @app.post("/api/concepts/{concept_id}/evaluation")
    async def api_submit_evaluation(concept_id: str, request: Request):
        _known(concept_id)
        form = await request.form()
        responses = _form_responses(form, service.get_activity(concept_id).get("questions", []))

        def work(job):
            result = service.submit_initial_evaluation(concept_id, responses)
            job.finish(
                f"/concepts/{concept_id}",
                {"card": "evaluation", "gap": result.get("gap")},
            )

        return _start(
            "grading",
            [("reading", "Lecture des réponses"), ("grading", "Correction"), ("levels", "Calcul des niveaux")],
            work,
            concept_id,
        )

    @app.post("/api/concepts/{concept_id}/lesson")
    def api_next_lesson(concept_id: str):
        _known(concept_id)

        def work(job):
            service.next_lesson(concept_id)
            job.finish(f"/concepts/{concept_id}")

        return _start(
            "lesson",
            [("planning", "Choix du prochain concept"), ("lesson", "Rédaction du cours")],
            work,
            concept_id,
        )

    @app.post("/api/concepts/{concept_id}/exercise")
    def api_next_exercise(concept_id: str):
        _known(concept_id)

        def work(job):
            service.next_exercise(concept_id)
            job.finish(f"/concepts/{concept_id}")

        return _start(
            "exercise",
            [("planning", "Préparation"), ("exercise", "Création de l'exercice")],
            work,
            concept_id,
        )

    @app.post("/api/concepts/{concept_id}/exercise/submit")
    async def api_submit_exercise(concept_id: str, request: Request):
        _known(concept_id)
        form = await request.form()
        text = "" if form.get("dont_know") else (form.get("response_text") or "")

        def work(job):
            result = service.submit_exercise(concept_id, text)
            job.finish(
                f"/concepts/{concept_id}",
                {
                    "card": "exercise",
                    "score": result.get("score"),
                    "feedback": result.get("feedback"),
                    "level": result.get("level"),
                    "level_before": result.get("level_before"),
                    "concept_label": result.get("concept_label"),
                    "progress": result.get("progress"),
                },
            )

        return _start(
            "grading",
            [("reading", "Lecture de la réponse"), ("grading", "Correction"), ("levels", "Mise à jour du niveau")],
            work,
            concept_id,
        )

    @app.post("/api/concepts/{concept_id}/final-exam")
    def api_start_final_exam(concept_id: str):
        _known(concept_id)

        def work(job):
            service.start_final_exam(concept_id)
            job.finish(f"/concepts/{concept_id}")

        return _start("exam", [("exam", "Rédaction de l'examen final")], work, concept_id)

    @app.post("/api/concepts/{concept_id}/final-exam/submit")
    async def api_submit_final_exam(concept_id: str, request: Request):
        _known(concept_id)
        form = await request.form()
        responses = _form_responses(form, service.get_activity(concept_id).get("questions", []))

        def work(job):
            service.submit_final_exam(concept_id, responses)
            job.finish(f"/concepts/{concept_id}")

        return _start(
            "grading",
            [("reading", "Lecture des réponses"), ("grading", "Correction"), ("levels", "Bilan final")],
            work,
            concept_id,
        )

    return app
