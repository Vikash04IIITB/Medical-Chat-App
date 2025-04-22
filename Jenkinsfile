pipeline {
    agent any

    environment {
        SUDO_PASSWORD = credentials('vikash-sudo') // ✅ Sudo credentials
        // AWS_ACCESS_KEY_ID = credentials('vikash-aws-access') // ✅ Commented AWS credentials
        // AWS_SECRET_ACCESS_KEY = credentials('vikash-aws-secret') // ✅ Commented AWS credentials
        // AWS_DEFAULT_REGION = "eu-north-1" // ✅ Commented AWS region
        // S3_BUCKET_NAME = "healthcarechatbot1" // ✅ Commented S3 bucket name
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'master', url: 'https://github.com/Vikash04IIITB/Medical-Chat-App.git'

            }
        }

        stage('Test Model') {
            steps {
                sh 'pwd'
                sh 'python3 training/test.py'
            }
        }

        stage('Train Model') {
            steps {
                script {
                    def trainImage = docker.build("vikash04/train-model:latest", '-f training/Dockerfile training')
                    withDockerRegistry([credentialsId: "DockerHubCred", url: ""]) {
                        trainImage.push("${env.BUILD_NUMBER}")
                    }
                }
            }
        }

        // Commented out AWS upload to S3 stage
        // stage('Upload to S3') {
        //     steps {
        //         sh """
        //             aws s3 cp ExtraTrees s3://${S3_BUCKET_NAME}/ExtraTrees --region ${AWS_DEFAULT_REGION}
        //         """
        //     }
        // }

        stage('Build Docker Frontend Image') {
            steps {
                dir('healthcare_chatbot_frontend') {
                    script {
                        frontendImage = docker.build("vikash04/react-app:frontend1")
                    }
                }
            }
        }

        stage('Push Docker Frontend Image') {
            steps {
                script {
                    withDockerRegistry([credentialsId: "DockerHubCred", url: ""]) {
                        frontendImage.push()
                    }
                }
            }
        }

        stage('Build Docker Backend Image') {
            steps {
                dir('healthcare_chatbot_backend') {
                    script {
                        backendImage = docker.build("vikash04/flask-app:backend1")
                    }
                }
            }
        }

        stage('Push Docker Backend Image') {
            steps {
                script {
                    withDockerRegistry([credentialsId: "DockerHubCred", url: ""]) {
                        backendImage.push()
                    }
                }
            }
        }

        stage('Deploy with Ansible') {
            steps {
                script {
                    withEnv(["SUDO_PASSWORD=${SUDO_PASSWORD}"]) {
                        ansiblePlaybook becomeUser: null, 
                                        colorized: true, 
                                        disableHostKeyChecking: true, 
                                        installation: 'Ansible', 
                                        inventory: './ansible-deploy/inventory', 
                                        playbook: './ansible-deploy/ansible-book.yml', 
                                        sudoUser: null,
                                        extraVars: [ansible_become_pass: SUDO_PASSWORD]
                    }
                }
            }
        }
    }
}
