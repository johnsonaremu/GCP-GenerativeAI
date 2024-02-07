from typing import Dict, List, Optional, Union
from google.cloud import aiplatform_v1beta1 as aiplatform
from google.api_core.client_options import ClientOptions


def get_multimodal_embeddings(
    project_id: str,
    location: str,
    text: Optional[str] = None,
    image_gcs_uri: Optional[str] = None,
    video_gcs_uri: Optional[str] = None,
) -> Dict[str, List]:
    client = aiplatform.PredictionServiceClient(
        client_options=ClientOptions(
            api_endpoint=f"{location}-aiplatform.googleapis.com"
        )
    )
    model = f"projects/{project_id}/locations/{location}/publishers/google/models/multimodalembedding@001"

    instance: Dict[str, Union[str, Dict[str, str]]] = {}
    if text:
        instance["text"] = text
    if image_gcs_uri:
        instance["image"] = {"gcsUri": image_gcs_uri}
    if video_gcs_uri:
        instance["video"] = {"gcsUri": video_gcs_uri}

    response = client.predict(
        endpoint=model,
        instances=[instance],
    )

    return {
        "text_embedding": response.predictions[0].get("textEmbedding"),
        "image_embedding": response.predictions[0].get("imageEmbedding"),
        # The video embeddings are chunked with offsets
        "video_embeddings": [
            {
                "start_offset_sec": ve.get("startOffsetSec"),
                "end_offset_sec": ve.get("endOffsetSec"),
                "embedding": ve.get("embedding"),
            }
            for ve in response.predictions[0].get("videoEmbeddings")
        ],
    }


PROJECT_ID = "cloud-llm-preview2"
LOCATION = "us-central1"
VIDEO_URI = "gs://ucs-demo/multimodal-embeddings/Video_20240206_120526_976.mp4"

embeddings = get_multimodal_embeddings(
    project_id=PROJECT_ID,
    location=LOCATION,
    video_gcs_uri=VIDEO_URI,
)

print(embeddings)
