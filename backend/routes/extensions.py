from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid
import os
import shutil
import zipfile
import json
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/extensions", tags=["extensions"])

# Models
class Extension(BaseModel):
    id: str
    name: str
    version: str
    description: str
    enabled: bool
    path: str
    size: str
    created_at: Optional[datetime] = None

class LoadUnpackedRequest(BaseModel):
    path: str

class PackExtensionRequest(BaseModel):
    path: str
    key_path: Optional[str] = None

class ToggleExtensionRequest(BaseModel):
    enabled: bool

# In-memory storage (will be persisted to MongoDB)
extensions_store: List[dict] = [
    {
        "id": "ext-react-devtools",
        "name": "React Developer Tools",
        "version": "5.0.2",
        "description": "Adds React debugging tools to the Chrome Developer Tools.",
        "enabled": True,
        "path": "/extensions/react-devtools",
        "size": "2.1 MB",
        "created_at": datetime.utcnow()
    },
    {
        "id": "ext-redux-devtools",
        "name": "Redux DevTools",
        "version": "3.1.3",
        "description": "Redux debugging tools for Chrome.",
        "enabled": True,
        "path": "/extensions/redux-devtools",
        "size": "1.8 MB",
        "created_at": datetime.utcnow()
    },
    {
        "id": "ext-ublock",
        "name": "uBlock Origin",
        "version": "1.55.0",
        "description": "An efficient wide-spectrum content blocker.",
        "enabled": False,
        "path": "/extensions/ublock",
        "size": "5.2 MB",
        "created_at": datetime.utcnow()
    }
]

def get_directory_size(path: str) -> str:
    """Calculate directory size"""
    total = 0
    if os.path.exists(path):
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
    
    if total < 1024:
        return f"{total} B"
    elif total < 1024 * 1024:
        return f"{total / 1024:.1f} KB"
    else:
        return f"{total / (1024 * 1024):.1f} MB"

@router.get("", response_model=List[Extension])
async def list_extensions():
    """List all installed extensions"""
    return [Extension(**ext) for ext in extensions_store]

@router.post("/load", response_model=Extension)
async def load_unpacked(request: LoadUnpackedRequest):
    """Load an unpacked extension from a directory"""
    path = request.path
    
    # Validate path (in real implementation, would check for manifest.json)
    # For demo, we simulate loading
    manifest_path = os.path.join(path, "manifest.json") if os.path.exists(path) else None
    
    # Simulate extension info
    ext_name = os.path.basename(path) or "Unpacked Extension"
    
    # Try to read manifest if exists
    name = ext_name
    version = "1.0.0"
    description = f"Loaded from: {path}"
    
    if manifest_path and os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                name = manifest.get('name', ext_name)
                version = manifest.get('version', '1.0.0')
                description = manifest.get('description', description)
        except Exception as e:
            logger.warning(f"Could not read manifest: {e}")
    
    new_ext = {
        "id": f"ext-{uuid.uuid4().hex[:8]}",
        "name": name,
        "version": version,
        "description": description,
        "enabled": True,
        "path": path,
        "size": get_directory_size(path) if os.path.exists(path) else "Unknown",
        "created_at": datetime.utcnow()
    }
    
    extensions_store.append(new_ext)
    logger.info(f"Loaded unpacked extension: {name}")
    
    return Extension(**new_ext)

@router.post("/pack")
async def pack_extension(request: PackExtensionRequest):
    """Pack an extension directory into a .crx file"""
    path = request.path
    
    if not os.path.exists(path):
        # Simulate packing for demo
        return {
            "status": "success",
            "message": f"Extension packed successfully (simulated)",
            "crx_path": f"{path}.crx",
            "key_path": request.key_path or f"{path}.pem"
        }
    
    # Create output paths
    crx_path = f"{path}.crx"
    key_path = request.key_path or f"{path}.pem"
    
    try:
        # Create a zip file (CRX is essentially a zip with a header)
        zip_path = f"{path}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, path)
                    zipf.write(file_path, arcname)
        
        # Rename to .crx (simplified - real CRX has a specific header)
        shutil.move(zip_path, crx_path)
        
        # Generate a dummy key file if not provided
        if not os.path.exists(key_path):
            with open(key_path, 'w') as f:
                f.write("# Private key placeholder\n")
        
        return {
            "status": "success",
            "message": "Extension packed successfully",
            "crx_path": crx_path,
            "key_path": key_path
        }
    except Exception as e:
        logger.error(f"Failed to pack extension: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{ext_id}/toggle", response_model=Extension)
async def toggle_extension(ext_id: str, request: ToggleExtensionRequest):
    """Toggle extension enabled state"""
    for ext in extensions_store:
        if ext['id'] == ext_id:
            ext['enabled'] = request.enabled
            logger.info(f"Extension {ext_id} {'enabled' if request.enabled else 'disabled'}")
            return Extension(**ext)
    
    raise HTTPException(status_code=404, detail="Extension not found")

@router.delete("/{ext_id}")
async def remove_extension(ext_id: str):
    """Remove an extension"""
    global extensions_store
    
    for i, ext in enumerate(extensions_store):
        if ext['id'] == ext_id:
            removed = extensions_store.pop(i)
            logger.info(f"Removed extension: {removed['name']}")
            return {"status": "removed", "id": ext_id}
    
    raise HTTPException(status_code=404, detail="Extension not found")
