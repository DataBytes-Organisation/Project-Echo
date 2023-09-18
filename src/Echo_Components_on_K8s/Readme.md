# Echo on Kubernetes 🔥

Welcome to this repo! This repository is designed to help anyone interested in learning about the DevOps side of projects, mainly Ops side (Operations). Whether you're a seasoned DevOps engineer or just starting out, you'll find this repo valuable to enhance your software delivery skills.

> **Note**: These tasks require individuals who are willing to dive into the codebase, understand how the components interact, and have the patience and critical thinking skills to troubleshoot and experiment. Whether you're a seasoned senior or an ambitious junior, this repository offers a chance to learn and work independently. There might not be immediate assistance available, so be prepared to explore and learn on your own.

## Objective of this Repository

The components built before were very tightly coupled, which means that there was never given a thought of how we would serve the echo frontend over the internet, for the end user. How will components scale, and how will debugging be done? It was all set up to be run on your PC. The aim of this folder is to provide a decoupled way to think of servers as entities, which are environment-dependent. What if the database server endpoint changes during production? Would you stop everything, start over, and go into the code to hardcode stuff? No, we use configurations that are provided wherever the component is running. Asking these questions is crucial in the real world.

Imagine adding new features to the front end and all you have to do is commit to the branch. The rest is automation. What about releasing versions of your software?

This folder is set up in a way that things are entities here, and things are loosely coupled here.

## Skills you can learn here:

- **Docker**: Containerization platform for building, shipping, and running applications.
- **Continuous Integration/Continuous Deployment (CI/CD)**: Automating the software delivery process for faster and more reliable releases.
- **Kubernetes**: Container orchestration tool for automating the deployment, scaling, and management of containerized applications.
- **Infrastructure as Code (IaC)**: Managing and provisioning infrastructure using code, often with tools like Terraform.
- **Configuration Management**: Tools like Ansible or Puppet for automating the configuration of servers and applications.
- **Scripting Skills**: Proficiency in writing scripts (Bash, PowerShell, etc.) for automation and customization.

## Important Links:

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Official Documentation](https://kubernetes.io/docs/home/)
- [Terraform Documentation](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [DevOps Institute](https://devopsinstitute.com/)
- [Github Actions Documentation](https://docs.github.com/en/actions)
- [CI/CD Explained](https://www.redhat.com/en/topics/devops/what-is-ci-cd)

## Kubernetes Landscape Terminology:

- **Pods**: The smallest deployable units in Kubernetes.
- **Deployment**: A Kubernetes object that manages a replicated application.
- **Service**: Provides a stable endpoint for accessing a set of pods.
- **Ingress**: Manages external access to the services.
- **ConfigMap and Secret**: Store configuration data separately from application code.
- **Namespace**: A way to divide cluster resources between multiple users.

## What's Already Done:

- Separated three components from the EchoNet (Fronted, Api and The database), making them dynamically configurable for different environments.
- Created a .bat script (start_echo_network.bat) for running the EchoNet locally, providing a convenient way for Windows users.
- Set up Kubernetes configurations for deploying components on cloud platforms.

## Todo List - don't get limited to this list, explore and expreriment:

- Create equivalent scripts for Mac and Linux users to run the EchoNet locally.
- Configure additional components like Engine, Model, and Simulator to integrate with the frontend API and database.
- Explore options for implementing CI/CD using tools like Terraform, Ansible, GitHub Actions, or Cloud Build.

Feel free to explore the existing projects and dive into the various branches to see different stages of development. Happy learning!

### Recommendations:

- **Use Visual Studio Code**:

  - We recommend using Visual Studio Code as your code editor for an optimized development experience.

- **Install Docker Extension**:
  - Consider installing the Docker extension for Visual Studio Code to streamline Docker-related tasks.

## Testing Locally

To test the components locally, follow these steps:

1. **Clone this Repository**:

   - Move to this repository on your local machine.

2. **Start Docker Engine**:

   - Ensure you have Docker installed and running on your system.

3. **Run the EchoNet**:
   - Execute the `start_echo_network.bat` file in your preferred command-line interface.

Note: To stop the containers and network, you can execute the `stop_echo_network.bat`. Also, While running these scripts, docker tend to sometimes create dangling images, which consume a lot of disk space sometimes. So keep checking for that.

Happy testing!
