'''import json
import random
import uuid

OUTPUT_FILE = "rag_stress_dataset.jsonl"

NUM_SCENARIOS = 100   # change to 500 if needed


components = {
    "Authentication Service": {
        "issue": "Users are logged out immediately after login.",
        "manual": "Ensure SESSION_TIMEOUT is set to at least 300 seconds in auth-config.yaml.",
        "logs": "Restarting auth-service container and clearing Redis session cache temporarily resolved the issue.",
        "wiki": "Disable token validation middleware to prevent session expiry."
    },

    "Payment API": {
        "issue": "Payment API returns HTTP 503 during peak traffic.",
        "manual": "Increase worker_pool_size to at least 16 in payment-service.yaml.",
        "logs": "Support engineer cleared upstream Nginx cache and restarted API pods.",
        "wiki": "Disable API gateway rate limiting."
    },

    "Cache Service": {
        "issue": "Application returns stale data after updates.",
        "manual": "Enable CACHE_INVALIDATION_TOPIC for event-based cache invalidation.",
        "logs": "Manual Redis cache flush temporarily resolved the issue.",
        "wiki": "Disable caching in application settings."
    },

    "Database Cluster": {
        "issue": "Frequent database connection timeouts.",
        "manual": "Increase max_connections to at least 500 for production workloads.",
        "logs": "Reduced application connection pool size temporarily stabilized service.",
        "wiki": "Restart primary database node when timeouts occur."
    },

    "Search Service": {
        "issue": "Search results missing recent records.",
        "manual": "Ensure INDEX_REFRESH_INTERVAL is set below 60 seconds.",
        "logs": "Restarting Elasticsearch nodes temporarily restored search results.",
        "wiki": "Rebuild entire search index nightly."
    },

    "Message Queue": {
        "issue": "Queue processing backlog increasing rapidly.",
        "manual": "Scale message consumers horizontally when queue depth exceeds threshold.",
        "logs": "Queue purge was used to clear backlog.",
        "wiki": "Disable retry logic for failed messages."
    },

    "Monitoring Service": {
        "issue": "Metrics missing for certain microservices.",
        "manual": "Ensure metrics exporter is enabled in service configuration.",
        "logs": "Restarting Prometheus server reloaded missing targets.",
        "wiki": "Disable metrics collection to reduce overhead."
    },

    "Container Scheduler": {
        "issue": "Containers repeatedly crash with OOM errors.",
        "manual": "Increase memory limits in deployment.yaml.",
        "logs": "Support manually restarted pods every few hours.",
        "wiki": "Disable container memory limits."
    }
}


conflict_types = [
    "Deprecated Procedure",
    "Experimental Fix",
    "Version Drift",
    "Security Regression",
    "Partial Fix",
    "Infrastructure Change",
    "False Correlation",
    "Multi-Step Resolution",
    "Ambiguous Query"
]


noise_documents = [
    "Load balancer TLS termination configuration guide.",
    "Disaster recovery procedure for database backups.",
    "Monitoring alert escalation policy documentation.",
    "Kubernetes node maintenance instructions.",
    "CI/CD deployment pipeline configuration.",
    "SSL certificate renewal procedure.",
    "Internal coding standards documentation.",
    "Storage cluster backup guidelines."
]


difficulty_levels = ["easy", "medium", "hard"]


def generate_noise():
    return random.sample(noise_documents, random.randint(3,5))


def generate_scenario(index):
    component = random.choice(list(components.keys()))
    data = components[component]

    scenario = {
        "scenario_id": f"SCN-{index:03d}",
        "component": component,
        "difficulty": random.choice(difficulty_levels),

        "problem_description": data["issue"],

        "source_A_manual": data["manual"],
        "source_B_support_logs": data["logs"],
        "source_C_legacy_wiki": data["wiki"],

        "noise_documents": generate_noise(),

        "ground_truth": data["manual"],
        "correct_source": "A",

        "conflict_type": random.choice(conflict_types)
    }

    return scenario


def generate_dataset():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        for i in range(1, NUM_SCENARIOS + 1):

            scenario = generate_scenario(i)

            f.write(json.dumps(scenario) + "\n")

    print(f"Dataset generated: {OUTPUT_FILE}")
    print(f"Total scenarios: {NUM_SCENARIOS}")


if __name__ == "__main__":
    generate_dataset()

        '''
'''
def generate_legacy_wiki():
    lines = []
    
    # Header lines
    lines.append("# Source C: Legacy Wiki\n")
    lines.append("> **WARNING:** Contains older procedures that may now be deprecated or incorrect.\n")
    lines.append("\n")
    
    # --- 10% Similarity with `synthetic_troubleshooting_1000.jsonl` (100 lines) ---
    lines.append("## 1. Archived Troubleshooting Records (Deprecated)\n")
    for i in range(1, 96):
        lines.append(f"- {{\"record\": \"Problem: background worker jobs stuck in pending state affecting component {i}. Fix Applied: refactored the retry logic with exponential backoff. Outcome: workflow execution returned to expected behavior.\"}}\n")
    
    # --- 10% Similarity with `technical_manual_1000_lines.docx` (100 lines) ---
    lines.append("\n## 2. Legacy System Module Specifications\n")
    for i in range(1, 99):
        lines.append(f"- This section describes operational behavior, architecture considerations, and configuration aspects of legacy module {i}. The system is designed using a microservice-oriented architecture with horizontal scalability. Engineers should ensure that service dependencies are monitored and configuration drift is minimized. (Line {i})\n")
        
    # --- 80% Original content for "Legacy Wiki" (800 lines) ---
    lines.append("\n## 3. Deprecated Operational Procedures\n")
    
    # Fill the remaining lines up to exactly 1000 lines
    remaining_lines = 1000 - len(lines)
    
    for i in range(1, remaining_lines + 1):
        lines.append(f"{i}. **[DEPRECATED]** Legacy Procedure Step {i}: Restart the monolith instance via `/etc/init.d/legacy-service restart` and manually clear the `/var/cache/app` directory. *Note: Do not apply to containerized production environments.*\n")

    # Write to Markdown file
    with open("legacy_wiki.md", "w", encoding="utf-8") as f:
        f.writelines(lines)
        
    print(f"Successfully generated 'legacy_wiki.md' with {len(lines)} lines.")

if __name__ == "__main__":
    generate_legacy_wiki()
    '''
import json
import re

INPUT_FILE = "D:\Week-2\Dataset\support_logs\support_log.jsonl"
OUTPUT_FILE = "D:\Week-2\Dataset\support_logs\support_log-2.jsonl"

pattern = re.compile(
    r"Problem:\s*(.*?)\.\s*Fix Applied:\s*(.*?)\.\s*Outcome:\s*(.*)\."
)

structured_rows = []

with open(INPUT_FILE, "r") as f:
    for line in f:
        data = json.loads(line)

        text = data["record"]
        match = pattern.search(text)

        if match:
            problem, fix, outcome = match.groups()

            structured_rows.append({
                "problem_description": problem.strip(),
                "fix": fix.strip(),
                "outcome": outcome.strip()
            })

with open(OUTPUT_FILE, "w") as f:
    for row in structured_rows:
        f.write(json.dumps(row) + "\n")

print(f"Converted {len(structured_rows)} rows")