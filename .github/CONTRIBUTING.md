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
