pipeline {
    agent any

    environment {
        SUDO_PASSWORD = credentials('vikash-sudo')
        VENV_PATH = "${HOME}/.jenkins-venv/bin/activate"
        // AWS_ACCESS_KEY_ID = credentials('vikash-aws-access')
        // AWS_SECRET_ACCESS_KEY = credentials('vikash-aws-secret')
        // AWS_DEFAULT_REGION = "eu-north-1"
        // S3_BUCKET_NAME = "healthcarechatbot1"
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'master', url: 'https://github.com/Vikash04IIITB/Medical-Chat-App.git'
            }
        }

        stage('Install Python Dependencies') {
            steps {
                sh '''
                    source ${VENV_PATH}
                    pip install --upgrade pip
                    pip install -r healthcare_chatbot_backend/requirements.txt
                '''
            }
        }

        stage('Test Model') {
            steps {
                sh '''
                    source ${VENV_PATH}
                    set -e
                    python3 training/test.py
                '''
            }
        }

        stage('Train Model') {
            steps {
                script {
                    def trainImage = docker.build("vikash04/train-model:${env.BUILD_NUMBER}", '-f training/Dockerfile training')
                    withDockerRegistry([credentialsId: "DockerHubCred", url: ""]) {
                        trainImage.push("latest")
                        trainImage.push("${env.BUILD_NUMBER}")
                    }
                }
            }
        }

        stage('Build Docker Frontend Image') {
            steps {
                dir('healthcare_chatbot_frontend') {
                    script {
                        frontendImage = docker.build("vikash04/react-app:frontend-${env.BUILD_NUMBER}")
                    }
                }
            }
        }

        stage('Push Docker Frontend Image') {
            steps {
                script {
                    withDockerRegistry([credentialsId: "DockerHubCred", url: ""]) {
                        frontendImage.push("frontend-${env.BUILD_NUMBER}")
                    }
                }
            }
        }

        stage('Build Docker Backend Image') {
            steps {
                dir('healthcare_chatbot_backend') {
                    script {
                        backendImage = docker.build("vikash04/flask-app:backend-${env.BUILD_NUMBER}")
                    }
                }
            }
        }

        stage('Push Docker Backend Image') {
            steps {
                script {
                    withDockerRegistry([credentialsId: "DockerHubCred", url: ""]) {
                        backendImage.push("backend-${env.BUILD_NUMBER}")
                    }
                }
            }
        }

        stage('Deploy with Ansible') {
            steps {
                script {
                    withEnv(["SUDO_PASSWORD=${SUDO_PASSWORD}"]) {
                        ansiblePlaybook(
                            becomeUser: null, 
                            colorized: true, 
                            disableHostKeyChecking: true, 
                            installation: 'Ansible', 
                            inventory: './ansible-deploy/inventory', 
                            playbook: './ansible-deploy/ansible-book.yml', 
                            sudoUser: null,
                            extraVars: [ansible_become_pass: SUDO_PASSWORD]
                        )
                    }
                }
            }
        }

        stage('Push to GitHub') {
            steps {
                script {
                    withCredentials([usernamePassword(credentialsId: 'git-credentials', usernameVariable: 'GIT_USER', passwordVariable: 'GIT_PASS')]) {
                        sh '''
                            git config user.name "Vikash"
                            git config user.email "vikash.kumar@example.com"
                            git add .
                            git commit -m "Automated update after build ${env.BUILD_NUMBER}"
                            git push https://${GIT_USER}:${GIT_PASS}@github.com/Vikash04IIITB/Medical-Chat-App.git master
                        '''
                    }
                }
            }
        }
    }

    post {
        failure {
            echo "Build failed, check logs for details!"
        }
        always {
            echo "Cleaning up..."
        }
    }
}
