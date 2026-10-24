pipeline {
    agent any
    stages {
        stage('Install') {
            steps {
                sh 'ffmpeg -version && ffprobe -version'
                sh 'python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"'
            }
        }
        stage('Verify') {
            steps {
                sh '.venv/bin/ruff check videoqa tests scripts'
                sh '.venv/bin/python -m compileall -q videoqa'
                sh '.venv/bin/pytest --junitxml=reports/tests.xml'
            }
        }
        stage('Golden QA') {
            steps {
                sh '.venv/bin/python scripts/generate_golden.py --output golden_dataset/generated'
                // Synthetic anomalies are intentionally candidates; still visible in JUnit as skipped.
                script {
                    def status = sh(returnStatus: true, script: '.venv/bin/videoqa batch --manifest golden_dataset/generated/manifest.yaml --output reports/golden --suspect warn --unverifiable warn')
                    if (status != 1) { error("Golden anomaly run expected exit 1, got ${status}") }
                }
            }
        }
    }
    post {
        always {
            junit allowEmptyResults: true, testResults: 'reports/**/*.xml'
            archiveArtifacts allowEmptyArchive: true, artifacts: 'reports/**/*'
        }
    }
}
