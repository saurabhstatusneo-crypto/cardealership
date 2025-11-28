from __future__ import annotations
import os
import re
import sys
import pathlib
from typing import List, Dict, Optional

try:
    from groq import Groq
except ImportError:
    print("Please install groq → pip install groq")
    sys.exit(1)

# ------------------------
# Load API Key
# ------------------------
API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    API_KEY = ""

if not API_KEY:
    raise Exception("❌ Missing GROQ_API_KEY environment variable.")

groq_client = Groq(api_key=API_KEY)

# ============================================================
# Detection patterns
# ============================================================
DB_PATTERNS = [
    r"@Repository", r"@Entity", r"@Table", r"@Column", r"@Id",
    r"JpaRepository", r"CrudRepository", r"PagingAndSortingRepository",
    r"MongoRepository", r"CassandraRepository",
    r"JdbcTemplate", r"NamedParameterJdbcTemplate",
    r"EntityManager", r"Session", r"CriteriaQuery", r"TypedQuery",
    r"createQuery\(", r"createNativeQuery\("
]

JDBC_PATTERNS = [
    r"Connection\b", r"PreparedStatement\b", r"ResultSet\b",
    r"DriverManager", r"SQLException"
]

SQL_PATTERNS = [
    r"SELECT\s", r"INSERT\s", r"UPDATE\s", r"DELETE\s", r"WHERE\s", r"JOIN\s"
]

DEPENDENCY_PATTERNS = [
    r"@Autowired",
    r"@Inject",
    r"@Resource",
    r"private\s+final\s+[\w\<\>\.\[\]]+\s+\w+;",
    r"private\s+[\w\<\>\.\[\]]+\s+\w+;",
]

SPRING_SKIP_PATTERNS = (
    r"@RestController|@Controller|@Configuration|@ComponentScan|@SpringBootApplication"
)

# ============================================================
# Method extraction
# ============================================================
METHOD_SIG_RE = re.compile(
    r"(?P<modifiers>public|protected|private)\s+"
    r"(?P<return>[\w\<\>\[\]\.? ,]+?)\s+"
    r"(?P<name>\w+)\s*"
    r"\((?P<params>[^\)]*)\)\s*"
    r"(\{)",
    re.MULTILINE
)


def extract_methods_with_bodies(code: str) -> List[Dict]:
    methods = []
    for m in METHOD_SIG_RE.finditer(code):
        brace_pos = code.find("{", m.end(0) - 1)
        if brace_pos == -1:
            continue

        depth, end = 0, None
        i = brace_pos
        while i < len(code):
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
            i += 1

        body = code[brace_pos:end] if end else "{ }"
        methods.append({
            "modifiers": m.group("modifiers"),
            "return": m.group("return").strip(),
            "name": m.group("name"),
            "params": m.group("params").strip(),
            "body": body.strip(),
        })

    return methods


# ============================================================
# Classify and dependency detection
# ============================================================
def detect_constructor_injection(code: str) -> bool:
    class_name = re.search(r"class\s+(\w+)", code)
    if not class_name:
        return False
    c = class_name.group(1)
    ctor_pattern = re.compile(rf"public\s+{c}\s*\(([^)]*)\)", re.MULTILINE)
    match = ctor_pattern.search(code)
    if match:
        params = match.group(1)
        return bool(params.strip())
    return False


def detect_dependencies(code: str) -> bool:
    return any(re.search(p, code) for p in DEPENDENCY_PATTERNS)


def detect_db_type(code: str) -> Optional[str]:
    if any(re.search(p, code, re.IGNORECASE) for p in DB_PATTERNS):
        return "JPA"
    if any(re.search(p, code, re.IGNORECASE) for p in JDBC_PATTERNS):
        return "JDBC"
    if any(re.search(p, code, re.IGNORECASE) for p in SQL_PATTERNS):
        return "SQL"
    return None


def classify_class(code: str) -> tuple:
    db = detect_db_type(code)
    has_dep = detect_dependencies(code)
    ctor_dep = detect_constructor_injection(code)
    if db:
        return "DB", db
    if ctor_dep or has_dep:
        return "SERVICE", ""
    return "PURE", ""


# ============================================================
# Simple analyzers for prompting
# ============================================================
def summarize_method_logic(method: Dict) -> str:
    body = method["body"]
    notes = []
    if re.search(r"\bif\b", body):
        notes.append("conditionals")
    if re.search(r"\bswitch\b", body):
        notes.append("switch")
    if re.search(r"\bfor\b|\bwhile\b|\bdo\b", body):
        notes.append("loops")
    if re.search(r"throw\s+new", body):
        notes.append("throws")
    if re.search(r"\bnull\b", body):
        notes.append("null checks")
    if any(re.search(p, body) for p in DB_PATTERNS + JDBC_PATTERNS + SQL_PATTERNS):
        notes.append("DB interactions")
    if re.search(r"[A-Z]\w+\.", body):
        notes.append("dependency calls")
    return "; ".join(notes) if notes else "simple logic"


# ============================================================
# Dependencies extraction
# ============================================================
def extract_class_dependencies(code: str) -> List[str]:
    deps = set()
    # field injection detection: look for lines like "private SomeDep dep;"
    for match in re.finditer(r"(private|protected|public)\s+([\w\<\>\[\]]+)\s+(\w+)\s*;", code):
        type_name = match.group(2)
        # avoid primitive types and the class itself later
        if type_name not in ("int", "long", "double", "float", "boolean", "char", "byte", "short"):
            deps.add(type_name)
    # constructor params
    class_name = re.search(r"class\s+(\w+)", code)
    if class_name:
        c = class_name.group(1)
        ctor_match = re.search(rf"public\s+{c}\s*\(([^)]*)\)", code, re.MULTILINE)
        if ctor_match:
            params = ctor_match.group(1).split(",")
            for p in params:
                p = p.strip()
                if not p:
                    continue
                parts = p.split()
                if len(parts) >= 2:
                    deps.add(parts[0])
    return sorted(deps)


# ============================================================
# Prompt builder
# ============================================================
IMPORT_MAP = {
    "Test": "import org.junit.jupiter.api.Test;",
    "assertEquals": "import static org.junit.jupiter.api.Assertions.assertEquals;",
    "assertTrue": "import static org.junit.jupiter.api.Assertions.assertTrue;",
    "assertFalse": "import static org.junit.jupiter.api.Assertions.assertFalse;",
    "assertThrows": "import static org.junit.jupiter.api.Assertions.assertThrows;",
    "Mockito": "import org.mockito.Mockito;",
    "Mock": "import org.mockito.Mock;",
    "InjectMocks": "import org.mockito.InjectMocks;",
    "MockitoExtension": "import org.mockito.junit.jupiter.MockitoExtension;",
    "ExtendWith": "import org.junit.jupiter.api.extension.ExtendWith;",
    "Optional": "import java.util.Optional;",
    "List": "import java.util.List;",
}

DEFAULT_TEST_IMPORTS = [
    "import org.junit.jupiter.api.Test;",
    "import static org.junit.jupiter.api.Assertions.*;"
]


def build_prompt(java_code, class_name, package_name, class_type, db_type, methods):
    method_summary = "\n".join(
        f"- {m['modifiers']} {m['return']} {m['name']}({m['params']}): {summarize_method_logic(m)}"
        for m in methods
    )

    if class_type == "PURE":
        strategy = (
            "This is pure business logic. Do NOT use Mockito. Instantiate using new ClassName(). "
            "Test all public methods with positive, negative, and edge cases. Do NOT add notes."
        )
    elif class_type == "SERVICE":
        strategy = (
            "This class has dependencies. Use JUnit5 + Mockito for dependencies only. "
            "Do NOT mock the class under test itself. Use @Mock for dependencies and @InjectMocks for the class under test."
        )
    else:
        strategy = (
            f"This class interacts with {db_type} database. Mock repositories/JDBC templates only. "
            "Do NOT mock the class under test itself."
        )

    prompt = f"""
Generate a JUnit5 test class (NO markdown, NO notes).
Package: {package_name}.tests
Class: {class_name}Test

Method summaries:
{method_summary}

{strategy}

Source:
{java_code}
"""
    return prompt


# ============================================================
# Groq call + postprocessing
# ============================================================
def remove_code_comments(code: str) -> str:
    # remove code fences and comments produced by LLM
    code = re.sub(r"^```(?:java)?", "", code, flags=re.MULTILINE)
    code = re.sub(r"```", "", code)
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)
    # remove blank lines
    code = "\n".join([ln.rstrip() for ln in code.splitlines() if ln.strip()])
    return code


def ensure_pure_instantiation(code: str, class_name: str) -> str:
    """
    For PURE classes: remove any Mockito extension and mock annotations, then
    ensure the test class instantiates the class under test directly.
    """
    # remove @ExtendWith(MockitoExtension.class)
    code = re.sub(r"@ExtendWith\s*\(\s*MockitoExtension\.class\s*\)\s*", "", code)

    # remove fields annotated with @Mock that match class_name
    code = re.sub(rf"@Mock\s+private\s+{re.escape(class_name)}\s+\w+\s*;\s*", "", code)

    # remove @InjectMocks fields of same type
    code = re.sub(rf"@InjectMocks\s+private\s+{re.escape(class_name)}\s+\w+\s*;\s*", "", code)

    # if there's no direct instantiation, insert one after class declaration
    # find the test class opening (public class XTest {)
    class_open_re = re.compile(rf"(public\s+class\s+{re.escape(class_name)}Test\s*\{{)", re.MULTILINE)
    if class_open_re.search(code):
        # check if a field like "private final ClassName" exists
        if not re.search(rf"private\s+final\s+{re.escape(class_name)}\s+\w+\s*=", code):
            # insert instantiation after the opening brace line
            code = class_open_re.sub(
                r"\1\n\n    private final " + class_name + " underTest = new " + class_name + "();",
                code
            )
    return code


def add_missing_imports(code: str, class_name: str, package_name: str, class_type: str) -> str:
    """
    Build import block, dedupe, and prevent adding Mockito imports for pure classes.
    Also remove any existing import lines from code body to avoid duplicates.
    """
    # strip existing import lines from code body
    code_lines = code.splitlines()
    code_without_imports = "\n".join([ln for ln in code_lines if not ln.strip().startswith("import ")])
    code_body = code_without_imports.strip()

    needed = set()

    # always import the class under test (from original package)
    if package_name and class_name:
        # class under test normally lives in package_name (not tests subpackage)
        needed.add(f"import {package_name}.{class_name};")

    # scan code body for tokens and map to imports
    for key, imp in IMPORT_MAP.items():
        # skip Mockito imports for pure classes
        if key in ("Mock", "Mockito", "InjectMocks", "MockitoExtension", "ExtendWith") and class_type == "PURE":
            continue
        # add import if key appears as a word in code body
        if re.search(rf"\b{re.escape(key)}\b", code_body):
            needed.add(imp)

    # always add default test imports
    for imp in DEFAULT_TEST_IMPORTS:
        needed.add(imp)

    import_block = "\n".join(sorted(needed))

    pkg_line = f"package {package_name}.tests;" if package_name else ""
    final_parts = [p for p in (pkg_line, import_block, code_body) if p]
    final = "\n\n".join(final_parts)
    # tidy up
    final = "\n".join([ln.rstrip() for ln in final.splitlines() if ln.strip()])
    return final


def generate_test(java_code, class_name, package, class_type, db_type, methods):
    prompt = build_prompt(java_code, class_name, package, class_type, db_type, methods)

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content or ""
    content = remove_code_comments(content)
    # If the class is pure remove mocking and add direct instantiation
    if class_type == "PURE":
        content = ensure_pure_instantiation(content, class_name)

    # Add imports and return
    return add_missing_imports(content.strip(), class_name, package, class_type)


# ============================================================
# File processing & main
# ============================================================
def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def write_file(path: str, content: str):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def process_java_files(root_dir: str = "src/main/java"):
    print("Scanning for Java files...")

    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".java"):
                continue

            path = os.path.join(root, file)
            code = read_file(path)

            if re.search(SPRING_SKIP_PATTERNS, code):
                print(f"Skipping controller/config: {file}")
                continue

            class_name = file[:-5]
            pkg_match = re.search(r"package\s+([\w\.]+);", code)
            pkg = pkg_match.group(1) if pkg_match else ""

            class_type, db_type = classify_class(code)
            print(f"{class_name}: Type → {class_type} {db_type or ''}")

            methods = extract_methods_with_bodies(code)
            methods = [m for m in methods if m["modifiers"] == "public"]

            if not methods:
                print(f"No public methods found in {class_name}")
                continue

            test_code = generate_test(code, class_name, pkg, class_type, db_type, methods)

            output_dir = pathlib.Path("src/test/java") / pathlib.Path(*pkg.split("."), "tests")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"{class_name}Test.java"
            write_file(str(output_file), test_code)

            print(f"Created test: {output_file}")

    print("\nAll test classes generated.")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "src/main/java"
    process_java_files(src)