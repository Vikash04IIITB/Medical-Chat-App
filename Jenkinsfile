pipeline {
    agent any

    environment {
        // Retrieve the sudo password from Jenkins credentials and set it as an environment variable
        SUDO_PASSWORD = credentials('SUDO_PASSWORD')
        AWS_ACCESS_KEY_ID = credentials('aws-access-key-id')
        AWS_SECRET_ACCESS_KEY = credentials('aws-secret-access-key')
        AWS_DEFAULT_REGION = "eu-north-1"
        S3_BUCKET_NAME = "healthcarechatbot1"
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/sankalp137rai/SPE_final_project.git'
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
                    def trainImage = docker.build("sankalp137rai/train-model:latest", '-f training/Dockerfile training')
                    withDockerRegistry([credentialsId: "DockerHubCred", url: ""]) {
                        trainImage.push("${env.BUILD_NUMBER}")
                    }

                }
            }
        }

        stage('Upload to S3') {
            steps {
                script {
                    sh """
                        aws s3 cp ExtraTrees s3://${S3_BUCKET_NAME}/ExtraTrees --region ${AWS_DEFAULT_REGION}
                    """
                }
            }
        }

        stage('Build Docker Frontend Image') {
            steps {
                dir('healthcare_chatbot_frontend') {
                    script {
                        frontendImage = docker.build("sankalp137rai/react-app:frontend1")
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
                        backendImage = docker.build("sankalp137rai/flask-app:backend1")
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
        // stage('Deploy with Ansible') {
        //     steps {
        //         script {
        //             ansiblePlaybook becomeUser: null, colorized: true, disableHostKeyChecking: true, installation: 'Ansible', inventory: 'ansible-deploy/inventory',
        //             playbook: './ansible-deploy/ansible-book.yml', sudoUser: null
        //         }
        //     }
        // }

        // stages {
        //         stage('Deploy to Minikube using ansible') {
        //             steps {
        //                 script {
        //                     // Export the sudo password and run the Ansible playbook
        //                     sh '''
        //                         export SUDO_PASSWORD=${SUDO_PASSWORD}
        //                         ansible-playbook /home/sankalp/Documents/healthcareChatbot/ansible-deploy/ansible-book.yml -i /home/sankalp/Documents/healthcareChatbot/ansible-deploy/inventory
        //                     '''
        //                 }
        //             }
        //         }
        //     }
        // }
        //
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

        // stage('Deploy frontend to Kubernetes') {
        //     steps {
        //         dir('kubernates') {
        //             script {
        //                 sh "kubectl apply -f frontend.yaml"
        //             }
        //         }
        //     }
        // }
    }
}
