"""FastAPI interface for the tutor.

The interface is intentionally simple: a welcome page listing the concepts in
progress, a concept page with the progress bar and the concept graph, a page
to deposit documents and start a new concept, and a settings page. Every
learning-path decision is delegated to the backend engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import Settings, load_settings
from .graphview import layout_graph
from .service import TutorService


PHASE_LABELS = {
    "not_started": "Non démarré",
    "initial_evaluation": "Évaluation initiale",
    "learning": "Apprentissage",
    "final_exam": "Examen final",
    "completed": "Terminé",
}


def phase_label(phase: str) -> str:
    return PHASE_LABELS.get(phase, phase)


def _redirect(concept_id: str, error: str | None = None) -> RedirectResponse:
    url = f"/concepts/{concept_id}"
    if error:
        url += f"?error={quote_plus(error)}"
    return RedirectResponse(url, status_code=303)


def _preferences(
    format_preference: str,
    focus: str,
    difficulties: str,
    daily_minutes: str,
) -> dict[str, Any]:
    preferences: dict[str, Any] = {"format": format_preference}
    if focus.strip():
        preferences["focus"] = focus.strip()
    if difficulties.strip():
        preferences["difficulties"] = difficulties.strip()
    if daily_minutes.strip():
        try:
            preferences["daily_minutes"] = int(daily_minutes)
        except ValueError:
            raise ValueError("le temps disponible doit être un nombre entier")
    return preferences


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    service = TutorService(settings)

    app = FastAPI(title="HermesTeacher")
    app.state.service = service
    app.state.settings = settings

    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    templates.env.globals["phase_label"] = phase_label
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
                "daily_minutes": "20",
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
        daily_minutes: str = Form("20"),
        eta: str = Form("0.3"),
        theta: str = Form("0.75"),
        files: list[UploadFile] = File(default=[]),
    ):
        try:
            eta_value = float(eta)
            theta_value = float(theta)
            preferences = _preferences(
                format_preference, focus, difficulties, daily_minutes
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
                    "daily_minutes": daily_minutes,
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
        concept = service.get_concept(concept_id)
        if concept is None:
            raise HTTPException(status_code=404, detail="concept not found")
        activity = service.get_activity(concept_id)
        graph_layout = None
        remaining_labels: list[str] = []
        if concept["graph"]:
            graph_layout = layout_graph(
                concept["graph"]["concepts"],
                concept["graph"]["edges"],
                concept["levels"],
            )
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
                "graph_layout": graph_layout,
                "remaining_labels": remaining_labels,
                "error": error,
            },
        )

    # ------------------------------------------------------------------
    # initial evaluation
    # ------------------------------------------------------------------
    @app.post("/concepts/{concept_id}/evaluation")
    async def submit_evaluation(concept_id: str, request: Request):
        try:
            activity = service.get_activity(concept_id)
            form = await request.form()
            responses = [
                {
                    "question_id": question["question_id"],
                    "response_text": form.get(
                        f"response-{question['question_id']}", ""
                    ),
                }
                for question in activity.get("questions", [])
            ]
            service.submit_initial_evaluation(concept_id, responses)
        except Exception as error:
            return _redirect(concept_id, str(error))
        return _redirect(concept_id)

    # ------------------------------------------------------------------
    # learning activities
    # ------------------------------------------------------------------
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
        try:
            form = await request.form()
            service.submit_exercise(
                concept_id, form.get("response_text", "")
            )
        except Exception as error:
            return _redirect(concept_id, str(error))
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
        try:
            activity = service.get_activity(concept_id)
            form = await request.form()
            responses = [
                {
                    "question_id": question["question_id"],
                    "response_text": form.get(
                        f"response-{question['question_id']}", ""
                    ),
                }
                for question in activity.get("questions", [])
            ]
            service.submit_final_exam(concept_id, responses)
        except Exception as error:
            return _redirect(concept_id, str(error))
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
                form.get("daily_minutes", ""),
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

    return app
