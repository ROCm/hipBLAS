## How to contribute

Our code contriubtion guidelines closely follows the model of [GitHub pull-requests](https://help.github.com/articles/using-pull-requests/). The hipBLAS repository follows a workflow which dictates a /master branch where releases are cut, and a
/develop branch which serves as an integration branch for new code.

## Pull-request guidelines
* target the **develop** branch for integration
* ensure code builds successfully.
* do not break existing test cases
* new functionality will only be merged with new unit tests
  * new unit tests should integrate within the existing googletest framework.
  * tests must have good code coverage
  * code must also have benchmark tests, and performance must approach the compute bound limit or memory bound limit.

## StyleGuide
This project follows the **CPP Core guidelines**, with few modifications or additions noted below.  All pull-requests should in good faith attempt to follow the guidelines stated therein, but we recognize that the content is lengthy.  Below we list our primary concerns when reviewing pull-requests.

### Interface
-  All public APIs are C89 compatible; all other library code should use c++14
  - Our minimum supported compiler is clang 3.6
-  Avoid CamelCase
  - This rule applies specifically to publicly visible APIs, but is also encouraged (not mandated) for internal code

### Philosophy
-  **P.2**: Write in ISO Standard C++ (especially to support windows, linux and macos plaforms )
-  **P.5**: Prefer compile-time checking to run-time checking

### Implementation
-  **SF.1**: Use a .cpp suffix for code files and .h for interface files if your project doesn't already follow another convention
  - We modify this rule:
    - .h: C header files
    - .hpp: C++ header files
-  **SF.5**: A .cpp file must include the .h file(s) that defines its interface
-  **SF.7**: Don't put a using-directive in a header file
-  **SF.8**: Use #include guards for all .h files
-  **SF.21**: Don't use an unnamed (anonymous) namespace in a header
-  **SL.10**: Prefer using STL array or vector instead of a C array
-  **C.9**: minimize exposure of members
-  **F.3**: Keep functions short and simple
-  **F.21**: To return multiple 'out' values, prefer returning a tuple
-  **R.1**: Manage resources automatically using RAII (this includes unique_ptr & shared_ptr)
-  **ES.11**:  use auto to avoid redundant repetition of type names
-  **ES.20**: Always initialize an object
-  **ES.23**: Prefer the {} initializer syntax
-  **ES.49**: If you must use a cast, use a named cast
-  **CP.1**: Assume that your code will run as part of a multi-threaded program
-  **I.2**: Avoid global variables
