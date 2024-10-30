# AI Video Editor

An AI-powered video editing application that automatically edits and combines video clips based on content analysis.

## Features
- Automatic video editing based on content analysis
- Multiple editing styles (Fast & Energized, Smooth & Professional, etc.)
- Cloud-based processing with GCP integration
- Real-time progress tracking

## Tech Stack
- Python/Flask backend
- OpenAI API for content analysis
- FFmpeg for video processing
- Google Cloud Platform
- Kubernetes for deployment

## Local Development
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate virtual environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create `.env` file with required environment variables
6. Run the application: `python app.py`

## Deployment
The application is deployed on Google Cloud Platform using:
- Google Kubernetes Engine (GKE)
- Cloud Storage
- Secret Manager

## Environment Variables
Required environment variables:
- `OPENAI_API_KEY`: OpenAI API key
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to GCP credentials file
- `PORT`: Application port (default: 8000)
