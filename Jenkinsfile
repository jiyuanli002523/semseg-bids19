#!/usr/bin/groovy

@Library(['github.com/indigo-dc/jenkins-pipeline-library']) _

pipeline {
    agent {
        label 'python'
    }

    environment {
        author_name = "G. Cavallaro (FZJ), M. Goetz (KIT), V. Kozlov (KIT)"
        author_email = "valentin.kozlov@kit.edu"
        app_name = "2dsemseg"
        job_location = "Pipeline-as-code/DEEP-OC-org/\
                        DEEP-OC-2dsemseg"
        job_result_url = ''
    }

    stages {
        stage('Fetch the repository') {
            steps {
                checkout scm
            }
        }
	
        stage('Style analysis: PEP8') {
            steps {
                ToxEnvRun('pep8')
            }
            post {
                always {
                    WarningsReport('Pep8')
                }
            }
        }

        stage('Style analysis: Pylint') {
            steps {
                ToxEnvRun('pylint')
            }
            post {
                always {
                    WarningsReport('Pylint')
                }
            }
        }

        stage("Re-build DEEP-OC-2dsemseg Docker image") {
            steps {
                script {
                    def job_result = JenkinsBuildJob("${env.job_location}")
                    def job_result_url = job_result.absoluteUrl
                }
            }
        }

        stage("Email notification") {
            steps {
                script {
                    def build_status =  currentBuild.result
                    build_status =  build_status ?: 'SUCCESS'
                    def subject = "New ${app_name} build in Jenkins@DEEP:\
                                   ${build_status}: Job '${env.JOB_NAME}\
                                   [${env.BUILD_NUMBER}]'"
                    def body = "Dear ${author_name},\nA new build of\
                                '${app_name}' DEEP application is available in\
                                Jenkins at:\n\n\t${env.BUILD_URL}\n\nterminated\
                                with '${build_status}' status.\n\nCheck console\
                                output at:\n\n\t${env.BUILD_URL}/console\n\n\
                                and resultant Docker image rebuilding job at:\
                                \n\n\t${job_result_url}\n\nDEEP Jenkins CI\
                                service"
                    EmailSend(subject, body, "${author_email}")
                }
            }
        }
    }
}
