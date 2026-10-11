"""Synthetic smoke-test data: controlled demographic variants (IPR-I Sec. 3.6).

Each qualification profile is rendered under several demographic variants while
the job-relevant content is IDENTICAL. The group code Z is a property of the
synthetic variant design (an experimental device); it is NOT an inference about
any real person. Relevance labels here are by construction (profile family =
job family) and exist only to exercise the MRR/Recall@k code.
THIS DATA CANNOT SUPPORT ANY CLAIM ABOUT BIAS OR MITIGATION."""

VARIANTS = [   # name, city, subject pronoun, possessive, Z (1 = reference, 0 = comparison)
    ("Rahul Sharma", "Delhi",   "He",  "his", 1),
    ("Amit Gupta",   "Mumbai",  "He",  "his", 1),
    ("Priya Verma",  "Lucknow", "She", "her", 0),
    ("Neha Singh",   "Pune",    "She", "her", 0),
]

PROFILES = {   # profile id -> (job family, body template)
    "sw1": ("software", "{name}\n{city}\n{S} built Python REST APIs with Docker and wrote unit tests. {P} projects used SQL databases."),
    "sw2": ("software", "{name}\n{city}\n{S} developed backend services in Python and Flask, automated testing and deployed with Docker. {P} work covered REST design."),
    "sw3": ("software", "{name}\n{city}\n{S} led API development in Python, wrote integration tests and maintained SQL schemas. {P} team shipped weekly."),
    "da1": ("data", "{name}\n{city}\n{S} built Excel and Tableau dashboards and ran SQL reporting queries. {P} analysis used statistics."),
    "da2": ("data", "{name}\n{city}\n{S} analysed sales data with SQL and statistics and presented dashboards. {P} reports guided decisions."),
    "da3": ("data", "{name}\n{city}\n{S} cleaned datasets, built Tableau dashboards and applied statistical tests. {P} SQL queries fed Excel models."),
}

JOBS = {
    "JD_software": "Software engineer needed: Python, REST APIs, Docker, unit testing and integration testing, SQL.",
    "JD_data": "Data analyst needed: SQL, Excel, Tableau dashboards, statistics and reporting.",
}


def build_dataset():
    resumes, groups, relevant = {}, {}, {"JD_software": set(), "JD_data": set()}
    for pid, (family, tpl) in PROFILES.items():
        for name, city, S, P, z in VARIANTS:
            rid = f"{pid}_{name.split()[0]}"
            resumes[rid] = tpl.format(name=name, city=city, S=S, P=P.capitalize())
            groups[rid] = z
            relevant["JD_software" if family == "software" else "JD_data"].add(rid)
    return resumes, JOBS, groups, relevant
