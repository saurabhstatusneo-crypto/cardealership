pipeline {
    agent any

    environment {
        // Jenkins Credentials
        GROQ_API_KEY     = credentials('GROQ_API_KEY')
        GITHUB_TOKEN     = credentials('GITHUB_TOKEN')
        GIT_HOME         = "/usr/bin/git"  // Default Git location in Ubuntu
        PATH = "${GIT_HOME}:${env.MAVEN_HOME}/bin:${env.JAVA_HOME}/bin:${env.PATH}"
    }

    stages {
        /* -----------------------------
           Load CI Config from properties file
        ------------------------------*/
        stage('Load CI Config') {
            steps {
                script {
                    def props = [:]
                    readFile('ci.properties').split('\n').each { line ->
                        line = line.trim()
                        if (line && !line.startsWith('#')) {
                            def (k, v) = line.split('=', 2)
                            props[k.trim()] = v.trim().replace('"','')
                        }
                    }

                    env.REPO_URL      = props['REPO_URL']
                    env.SOURCE_BRANCH = props['SOURCE_BRANCH']
                    env.TARGET_BRANCH = props['TARGET_BRANCH']
                    env.JAVA_HOME     = props['JAVA_HOME']
                    env.MAVEN_HOME    = props['MAVEN_HOME']
                    env.PYTHON_WIN    = props['PYTHON_WIN']
                    env.PATH          = props['PATH']
                }
            }
        }

        /* -----------------------------
           CHECKOUT
        ------------------------------*/
        stage('Checkout SCM') {
            steps {
                cleanWs()
                withCredentials([string(credentialsId: 'GITHUB_TOKEN', variable: 'TOKEN')]) {
                    sh '''
                        git --version
                        # Configure git globally (works even outside a repo)
                        git config --global user.name "saurabhstatusneo-crypto"
                        git config --global user.email "saurabhstatusneo@gmail.com"

                        # Clone if missing
                        if [ ! -d ".git" ]; then
                            git clone -b "$SOURCE_BRANCH" "https://$TOKEN@github.com/saurabhstatusneo-crypto/cardealership.git" .
                        fi

                        # Sync with remote
                        git fetch origin "$SOURCE_BRANCH"
                        git checkout "$SOURCE_BRANCH"
                        git reset --hard "origin/$SOURCE_BRANCH"
                        git clean -fd
                    '''
                }
            }
        }

        /* -----------------------------
           SETUP PYTHON + VENV
        ------------------------------*/
        stage('Setup Python & Venv') {
            steps {
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install -U pip wheel setuptools groq
                '''
            }
        }

        /* -----------------------------
           GENERATE TESTS
        ------------------------------*/
        stage('Generate Tests using AI') {
            when { expression { fileExists('generate_tests.py') } }
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            . venv/bin/activate
                            pip install groq
                            python generate_tests.py
                        '''
                    }
                }
            }
        }

        stage('Build and Run Tests') {
            when { expression { fileExists('pom.xml') } }
            steps {
                script {
                    if (isUnix()) {
                        sh """
                            export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
                            export PATH=\$JAVA_HOME/bin:\$PATH
                            java -version
                            javac -version
                            mvn clean install
                        """
                    }
                }
            }
        }

        /* -----------------------------
           GIT COMMIT & PUSH
        ------------------------------*/
        stage('Commit & Push') {
            steps {
                script {
                    sh '''
                        git add src/test
                        CHANGED=$(git status --porcelain)

                        if [ ! -z "$CHANGED" ]; then
                            git checkout -B $TARGET_BRANCH
                            git add src/test
                            git commit -m "chore: AI test cases updated [auto]" || echo "No changes to commit"
                            git push -u origin $TARGET_BRANCH --force
                        else
                            echo "No new generated test changes."
                        fi
                    '''
                }
            }
        }
    }

    post {
        always { echo "Pipeline finished" }
        success { echo "✔ SUCCESS" }
        failure { echo "❌ FAILURE" }
    }
}