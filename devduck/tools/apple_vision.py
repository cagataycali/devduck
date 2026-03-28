"""👁️ Apple Vision Intelligence - OCR, image analysis via Apple's Neural Engine."""

from strands import tool
from typing import Dict, Any, List


@tool
def apple_vision(
    action: str = "ocr",
    image_path: str = None,
    text: str = None,
    languages: List[str] = None,
) -> Dict[str, Any]:
    """👁️ On-device Vision AI via Apple Neural Engine - OCR, barcode, image analysis.

    All processing happens on-device using Apple's Neural Engine. Zero cloud calls.

    Args:
        action: Action to perform:
            - "ocr": Extract text from image (screenshot, photo, document)
            - "ocr_screen": OCR the current screen (takes screenshot first)
            - "barcode": Detect barcodes/QR codes in image
            - "faces": Detect faces and landmarks
            - "rectangles": Detect rectangles (documents, cards)
            - "horizon": Detect horizon angle
            - "saliency": Find attention-grabbing regions
            - "languages": List supported OCR languages
        image_path: Path to image file (png, jpg, heic, etc.)
        text: Not used (reserved for future text-in-image search)
        languages: OCR language hints (default: ['en-US'])

    Returns:
        Dict with vision analysis results
    """
    try:
        from Vision import (
            VNRecognizeTextRequest,
            VNImageRequestHandler,
            VNDetectBarcodesRequest,
            VNDetectFaceRectanglesRequest,
            VNDetectFaceLandmarksRequest,
            VNDetectRectanglesRequest,
            VNDetectHorizonRequest,
            VNGenerateAttentionBasedSaliencyImageRequest,
        )
        from Foundation import NSURL
        import Quartz

        if action == "languages":
            req = VNRecognizeTextRequest.alloc().init()
            langs, err = req.supportedRecognitionLanguagesAndReturnError_(None)
            return {
                "status": "success",
                "content": [{"text": f"Supported OCR languages:\n{chr(10).join(langs)}"}],
            }

        # Handle screen capture
        if action == "ocr_screen":
            import subprocess
            import tempfile
            import os

            image_path = os.path.join(tempfile.gettempdir(), "devduck_screen_ocr.png")
            subprocess.run(["screencapture", "-x", image_path], check=True)

        if not image_path:
            return {"status": "error", "content": [{"text": "image_path required (or use ocr_screen)"}]}

        # Load image
        import os
        image_path = os.path.expanduser(image_path)
        if not os.path.exists(image_path):
            return {"status": "error", "content": [{"text": f"File not found: {image_path}"}]}

        image_url = NSURL.fileURLWithPath_(image_path)
        handler = VNImageRequestHandler.alloc().initWithURL_options_(image_url, None)

        if action in ("ocr", "ocr_screen"):
            request = VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(0)  # 0 = accurate, 1 = fast
            request.setUsesLanguageCorrection_(True)

            if languages:
                request.setRecognitionLanguages_(languages)

            success = handler.performRequests_error_([request], None)

            results = request.results()
            if not results:
                return {"status": "success", "content": [{"text": "No text detected in image."}]}

            lines = []
            for obs in results:
                text_val = obs.topCandidates_(1)[0].string()
                confidence = obs.topCandidates_(1)[0].confidence()
                bbox = obs.boundingBox()
                lines.append({
                    "text": text_val,
                    "confidence": round(confidence, 3),
                    "y": round(bbox.origin.y, 3),
                })

            # Sort by Y position (top to bottom, inverted in Vision coords)
            lines.sort(key=lambda x: -x["y"])

            full_text = "\n".join(l["text"] for l in lines)
            detail = "\n".join(f"  [{l['confidence']:.0%}] {l['text']}" for l in lines)

            text_out = f"📝 OCR Result ({len(lines)} lines):\n\n{full_text}\n\n--- Detail ---\n{detail}"
            return {"status": "success", "content": [{"text": text_out}]}

        elif action == "barcode":
            request = VNDetectBarcodesRequest.alloc().init()
            handler.performRequests_error_([request], None)

            results = request.results()
            if not results:
                return {"status": "success", "content": [{"text": "No barcodes/QR codes found."}]}

            lines = [f"🔲 Found {len(results)} barcode(s):\n"]
            for obs in results:
                lines.append(f"  Type: {obs.symbology()}")
                lines.append(f"  Payload: {obs.payloadStringValue()}")
                lines.append(f"  Confidence: {obs.confidence():.0%}")
                lines.append("")

            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        elif action == "faces":
            # Detect faces with landmarks
            request = VNDetectFaceLandmarksRequest.alloc().init()
            handler.performRequests_error_([request], None)

            results = request.results()
            if not results:
                return {"status": "success", "content": [{"text": "No faces detected."}]}

            lines = [f"👤 Found {len(results)} face(s):\n"]
            for i, face in enumerate(results):
                bbox = face.boundingBox()
                lines.append(f"  Face {i+1}:")
                lines.append(f"    Position: ({bbox.origin.x:.2f}, {bbox.origin.y:.2f})")
                lines.append(f"    Size: {bbox.size.width:.2f} x {bbox.size.height:.2f}")
                lines.append(f"    Confidence: {face.confidence():.0%}")

                landmarks = face.landmarks()
                if landmarks:
                    has = []
                    if landmarks.leftEye():
                        has.append("left eye")
                    if landmarks.rightEye():
                        has.append("right eye")
                    if landmarks.nose():
                        has.append("nose")
                    if landmarks.outerLips():
                        has.append("mouth")
                    if landmarks.leftEyebrow():
                        has.append("eyebrows")
                    lines.append(f"    Landmarks: {', '.join(has)}")
                lines.append("")

            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        elif action == "rectangles":
            request = VNDetectRectanglesRequest.alloc().init()
            request.setMaximumObservations_(10)
            handler.performRequests_error_([request], None)

            results = request.results()
            if not results:
                return {"status": "success", "content": [{"text": "No rectangles detected."}]}

            lines = [f"📐 Found {len(results)} rectangle(s):\n"]
            for i, rect in enumerate(results):
                tl = rect.topLeft()
                tr = rect.topRight()
                bl = rect.bottomLeft()
                br = rect.bottomRight()
                lines.append(f"  Rect {i+1}: TL({tl.x:.2f},{tl.y:.2f}) BR({br.x:.2f},{br.y:.2f}) conf={rect.confidence():.0%}")

            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        elif action == "horizon":
            request = VNDetectHorizonRequest.alloc().init()
            handler.performRequests_error_([request], None)

            results = request.results()
            if not results:
                return {"status": "success", "content": [{"text": "No horizon detected."}]}

            angle = results[0].angle()
            import math
            degrees = math.degrees(angle)
            return {"status": "success", "content": [{"text": f"🌅 Horizon angle: {degrees:.2f}° (radians: {angle:.4f})"}]}

        elif action == "saliency":
            request = VNGenerateAttentionBasedSaliencyImageRequest.alloc().init()
            handler.performRequests_error_([request], None)

            results = request.results()
            if not results:
                return {"status": "success", "content": [{"text": "No saliency data."}]}

            salient = results[0].salientObjects()
            if salient:
                lines = [f"🎯 Found {len(salient)} attention region(s):\n"]
                for i, obj in enumerate(salient):
                    bbox = obj.boundingBox()
                    lines.append(f"  Region {i+1}: ({bbox.origin.x:.2f},{bbox.origin.y:.2f}) size={bbox.size.width:.2f}x{bbox.size.height:.2f} conf={obj.confidence():.0%}")
                return {"status": "success", "content": [{"text": "\n".join(lines)}]}

            return {"status": "success", "content": [{"text": "Saliency map generated (no discrete objects)"}]}

        else:
            return {"status": "error", "content": [{"text": f"Unknown action: {action}. Use: ocr, ocr_screen, barcode, faces, rectangles, horizon, saliency, languages"}]}

    except ImportError as e:
        return {"status": "error", "content": [{"text": f"Install: pip install pyobjc-framework-Vision pyobjc-framework-Quartz\nError: {e}"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error: {e}"}]}
