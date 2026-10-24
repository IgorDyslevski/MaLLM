from typing import Annotated, cast

from fastapi import APIRouter, Depends, Request
from mlflow.pyfunc import PyFuncModel
from starlette.concurrency import run_in_threadpool

from src.schemas import GenerationRequest, GenerationResult, InferenceErrorResponse
from src.services.inference import process

process_router = APIRouter(tags=["Inference"])


def loaded_model(request: Request) -> PyFuncModel:
    """Вернуть модель, загруженную при старте приложения."""
    return cast(PyFuncModel, request.state.model)


@process_router.post("/process", responses={503: {"model": InferenceErrorResponse}})
async def process_text(
    request: GenerationRequest,
    model: Annotated[PyFuncModel, Depends(loaded_model)],
) -> GenerationResult:
    return await run_in_threadpool(process, request, model)
