# Project Echo - Sprint Task Assignments

**Project:** Deploy live wildlife detection system with cost monitoring  
**Duration:** 2 Sprints (4 weeks)  
**Date:** November 20, 2025

---

## 👥 TEAM ROSTER

| Student ID | Level | Target Grade | Role | Team |
|------------|-------|--------------|------|------|
| 220618261 | Senior (Project B) | HD | Cloud Member 1 - Infrastructure Lead | Cloud |
| S224097689 | Senior (Project B) | HD | Cloud Member 2 - Deployment Lead | Cloud |
| S225158107 | Senior (Project B) | D | API Member 1 - Backend Lead | API |
| 223856998 | Junior (Project A) | D | API Member 2 - Frontend Lead | API |
| 224142778 | Senior (Project B) | Pass | Cloud Member 3 - Billing Support | Cloud |

**Assignment Strategy:**
- **HD Students (220618261, S224097689)**: Critical infrastructure & deployment tasks
- **D Students (S225158107, 223856998)**: Backend/Frontend development tasks
- **Pass Student (224142778)**: Support tasks with documentation focus

---

## 📋 QUICK VIEW TABLE

| Team Member | Student ID | Sprint | Priority | Task | Status |
|------------|------------|--------|----------|------|--------|
| **Cloud Member 1 (220618261)** | 220618261 | 1 | HIGH | Set up GKE cluster with node pools and networking | ⬜ |
| **Cloud Member 1 (220618261)** | 220618261 | 1 | HIGH | Configure VPC, firewall rules, and persistent storage | ⬜ |
| **Cloud Member 1 (220618261)** | 1 | MEDIUM | Deploy MongoDB StatefulSet with 1000+ detections | ⬜ |
| **Cloud Member 1 (220618261)** | 220618261 | 2 | MEDIUM | Set up Cloud Monitoring dashboards and uptime checks | ⬜ |
| **Cloud Member 1 (220618261)** | 220618261 | 2 | NORMAL | Configure auto-scaling policies for Engine pods | ⬜ |
| **Cloud Member 1 (220618261)** | 220618261 | 2 | NORMAL | Set up CI/CD pipeline with Cloud Build | ⬜ |
| **Cloud Member 2 (S224097689)** | S224097689 | 1 | HIGH | Push Docker images to Google Container Registry | ⬜ |
| **Cloud Member 2 (S224097689)** | S224097689 | 1 | HIGH | Deploy all 6 services to GKE using K8s configs | ⬜ |
| **Cloud Member 2 (S224097689)** | S224097689 | 1 | HIGH | Configure LoadBalancer and obtain external IPs | ⬜ |
| **Cloud Member 2 (S224097689)** | S224097689 | 2 | HIGH | Configure public domain and Cloud DNS | ⬜ |
| **Cloud Member 2 (S224097689)** | S224097689 | 2 | HIGH | Set up SSL certificates and HTTPS | ⬜ |
| **Cloud Member 2 (S224097689)** | S224097689 | 2 | NORMAL | Implement staging environment for testing | ⬜ |
| **Cloud Member 3 (224142778)** | 224142778 | 1 | MEDIUM | Initialize database with sample data | ⬜ |
| **Cloud Member 3 (224142778)** | 224142778 | 1 | NORMAL | Configure MongoDB backups to Cloud Storage | ⬜ |
| **Cloud Member 3 (224142778)** | 224142778 | 1 | NORMAL | Document deployment procedures | ⬜ |
| **Cloud Member 3 (224142778)** | 224142778 | 2 | HIGH | Enable GCP Billing API and create service account | ⬜ |
| **Cloud Member 3 (224142778)** | 224142778 | 2 | HIGH | Set up budgets and alert thresholds (50%, 80%, 100%) | ⬜ |
| **Cloud Member 3 (224142778)** | 224142778 | 2 | MEDIUM | Write BigQuery SQL queries for cost analytics | ⬜ |
| **API Member 1 (S225158107)** | S225158107 | 1 | HIGH | Update Engine to run 24/7 with continuous processing | ⬜ |
| **API Member 1 (S225158107)** | S225158107 | 1 | HIGH | Implement WebSocket/SSE for real-time detections | ⬜ |
| **API Member 1 (S225158107)** | S225158107 | 1 | MEDIUM | Deploy IoT simulator as 24/7 Kubernetes job | ⬜ |
| **API Member 1 (S225158107)** | S225158107 | 2 | HIGH | Create admin cost API endpoints (/admin/costs/*) | ⬜ |
| **API Member 1 (S225158107)** | S225158107 | 2 | HIGH | Implement JWT authentication for admin access | ⬜ |
| **API Member 1 (S225158107)** | S225158107 | 2 | MEDIUM | Add user roles and permission checking | ⬜ |
| **API Member 2 (223856998)** | 223856998 | 1 | HIGH | Build live detection map with real-time updates | ⬜ |
| **API Member 2 (223856998)** | 223856998 | 1 | HIGH | Display species markers with confidence >70% | ⬜ |
| **API Member 2 (223856998)** | 223856998 | 1 | MEDIUM | Show detection details (species, location, timestamp) | ⬜ |
| **API Member 2 (223856998)** | 223856998 | 2 | HIGH | Create admin cost dashboard HTML page | ⬜ |
| **API Member 2 (223856998)** | 223856998 | 2 | MEDIUM | Build interactive charts (pie, line, gauge) | ⬜ |
| **API Member 2 (223856998)** | 223856998 | 2 | NORMAL | Add date range picker and CSV export | ⬜ |
| **API Member 3 (UNASSIGNED)** | - | 1 | HIGH | Create detection storage/retrieval API endpoints | ⬜ |
| **API Member 3 (UNASSIGNED)** | - | 1 | MEDIUM | Implement pagination and filtering (species, date, location) | ⬜ |
| **API Member 3 (UNASSIGNED)** | - | 1 | NORMAL | Optimize MongoDB indexes for performance | ⬜ |
| **API Member 3 (UNASSIGNED)** | - | 2 | MEDIUM | Add admin budget control forms | ⬜ |
| **API Member 3 (UNASSIGNED)** | - | 2 | NORMAL | Implement "Pause Services" functionality | ⬜ |
| **API Member 3 (UNASSIGNED)** | - | 2 | NORMAL | Load test API for 100+ concurrent users | ⬜ |
| **API Member 3 (UNASSIGNED)** | - | 2 | NORMAL | Write admin documentation and demo video | ⬜ |

**Status Legend:** ⬜ Not Started | 🔄 In Progress | ✅ Complete | ⚠️ Blocked

**Current Blockers:**
- ⚠️ Terraform apply halted for `sit-23t1-project-echo-25288b9` until `roles/serviceusage.serviceUsageAdmin` and `roles/iam.serviceAccountAdmin` are granted to `s224097689@deakin.edu.au`.

**Note:** API Member 3 tasks need to be distributed among S225158107, 223856998, or 224142778 if needed.

---

## CLOUD TEAM (3 Members)

### **Cloud Member 1 - Infrastructure Lead (220618261 - HD Target)**
**Sprint 1:**
- [ ] **[HIGH]** Set up GKE cluster with node pools and networking
- [ ] **[HIGH]** Configure VPC, firewall rules, and persistent storage
- [ ] **[MEDIUM]** Deploy MongoDB StatefulSet with 1000+ detections

**Sprint 2:**
- [ ] **[MEDIUM]** Set up Cloud Monitoring dashboards and uptime checks
- [ ] **[NORMAL]** Configure auto-scaling policies for Engine pods
- [ ] **[NORMAL]** Set up CI/CD pipeline with Cloud Build

---

### **Cloud Member 2 - Deployment Lead (S224097689 - HD Target)**
**Sprint 1:**
- [ ] **[HIGH]** Push Docker images to Google Container Registry
- [ ] **[HIGH]** Deploy all 6 services to GKE using K8s configs
- [ ] **[HIGH]** Configure LoadBalancer and obtain external IPs

**Sprint 2:**
- [ ] **[HIGH]** Configure public domain and Cloud DNS
- [ ] **[HIGH]** Set up SSL certificates and HTTPS
- [ ] **[NORMAL]** Implement staging environment for testing

---

### **Cloud Member 3 - Billing Support (224142778 - Pass Target)**
**Sprint 1:**
- [ ] **[MEDIUM]** Initialize database with sample data
- [ ] **[NORMAL]** Configure MongoDB backups to Cloud Storage
- [ ] **[NORMAL]** Document deployment procedures

**Sprint 2:**
- [ ] **[HIGH]** Enable GCP Billing API and create service account
- [ ] **[HIGH]** Set up budgets and alert thresholds (50%, 80%, 100%)
- [ ] **[MEDIUM]** Write BigQuery SQL queries for cost analytics

---

## API TEAM (3 Members)

### **API Member 1 - Backend Lead (S225158107 - D Target)**
**Sprint 1:**
- [ ] **[HIGH]** Update Engine to run 24/7 with continuous processing
- [ ] **[HIGH]** Implement WebSocket/SSE for real-time detections
- [ ] **[MEDIUM]** Deploy IoT simulator as 24/7 Kubernetes job

**Sprint 2:**
- [ ] **[HIGH]** Create admin cost API endpoints (/admin/costs/*)
- [ ] **[HIGH]** Implement JWT authentication for admin access
- [ ] **[MEDIUM]** Add user roles and permission checking

---

### **API Member 2 - Frontend Lead (223856998 - Junior, D Target)**
**Sprint 1:**
- [ ] **[HIGH]** Build live detection map with real-time updates
- [ ] **[HIGH]** Display species markers with confidence >70%
- [ ] **[MEDIUM]** Show detection details (species, location, timestamp)

**Sprint 2:**
- [ ] **[HIGH]** Create admin cost dashboard HTML page
- [ ] **[MEDIUM]** Build interactive charts (pie, line, gauge)
- [ ] **[NORMAL]** Add date range picker and CSV export

---

### **UNASSIGNED TASKS (Can be distributed)**
**Sprint 1:**
- [ ] **[HIGH]** Create detection storage/retrieval API endpoints (Suggest: S225158107)
- [ ] **[MEDIUM]** Implement pagination and filtering (Suggest: 223856998)
- [ ] **[NORMAL]** Optimize MongoDB indexes (Suggest: 224142778)

**Sprint 2:**
- [ ] **[MEDIUM]** Add admin budget control forms (Suggest: 223856998)
- [ ] **[NORMAL]** Implement "Pause Services" functionality (Suggest: 224142778)
- [ ] **[NORMAL]** Load test API for 100+ concurrent users (Suggest: S225158107)
- [ ] **[NORMAL]** Write admin documentation and demo video (Suggest: 224142778)

---

## DELIVERABLES

### **Sprint 1 End (Week 2):**
- ✅ Live system deployed on GCP
- ✅ Public URL showing real-time wildlife detections
- ✅ Map updating every 1-5 minutes with new detections
- ✅ MongoDB with 1000+ detection records

### **Sprint 2 End (Week 4):**
- ✅ 24/7 monitoring with alert notifications
- ✅ HTTPS domain with SSL certificate
- ✅ Admin cost dashboard showing GCP spending
- ✅ Budget alerts configured and tested
- ✅ API documentation and demo videos

---

## DAILY STANDUP QUESTIONS
1. What did I complete yesterday?
2. What am I working on today?
3. Any blockers?

## SPRINT REVIEW
- **Sprint 1:** Demo live detection system
- **Sprint 2:** Demo admin cost dashboard

---

**Questions?** Contact project lead.
