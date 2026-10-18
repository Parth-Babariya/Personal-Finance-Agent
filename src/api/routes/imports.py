"""Bank statement import: upload a CSV, review the preview, then commit."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, status

from ...services.importer import MAX_BYTES, ImportService, StatementError
from ..deps import get_current_user_id, get_imports
from ..schemas import ImportCommit

router = APIRouter(prefix="/imports", tags=["imports"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_statement(
    file: UploadFile,
    imports: ImportService = Depends(get_imports),
    user_id: str = Depends(get_current_user_id),
) -> dict[str, Any]:
    """Parse and categorise a CSV statement. Nothing is saved until the preview is committed."""
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Upload a .csv file")
    content = await file.read(MAX_BYTES + 1)
    try:
        return await imports.preview(user_id, file.filename or "statement.csv", content)
    except StatementError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{import_id}")
def get_import(
    import_id: str,
    imports: ImportService = Depends(get_imports),
    user_id: str = Depends(get_current_user_id),
) -> dict[str, Any]:
    batch = imports.get(user_id, import_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Import not found")
    return batch


@router.post("/{import_id}/commit")
def commit_import(
    import_id: str,
    req: ImportCommit,
    imports: ImportService = Depends(get_imports),
    user_id: str = Depends(get_current_user_id),
) -> dict[str, Any]:
    try:
        result = imports.commit(user_id, import_id, [r.model_dump() for r in req.rows])
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Import not found")
    return result


@router.delete("/{import_id}", status_code=status.HTTP_204_NO_CONTENT)
def discard_import(
    import_id: str,
    imports: ImportService = Depends(get_imports),
    user_id: str = Depends(get_current_user_id),
) -> Response:
    if not imports.discard(user_id, import_id):
        raise HTTPException(status_code=404, detail="No pending import with this id")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
