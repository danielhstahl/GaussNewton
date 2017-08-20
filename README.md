| [Linux][lin-link] | [Windows][win-link] | [Codecov][cov-link] |
| :---------------: | :-----------------: | :-------------------: |
| ![lin-badge]      | ![win-badge]        | ![cov-badge]          |

[lin-badge]: https://travis-ci.org/phillyfan1138/GaussNewton.svg?branch=master "Travis build status"
[lin-link]:  https://travis-ci.org/phillyfan1138/GaussNewton "Travis build status"
[win-badge]: https://ci.appveyor.com/api/projects/status/qdyfevsgl7tvfyy8?svg=true "AppVeyor build status"
[win-link]:  https://ci.appveyor.com/project/phillyfan1138/gaussnewton "AppVeyor build status"
[cov-badge]: https://codecov.io/gh/phillyfan1138/GaussNewton/branch/master/graph/badge.svg
[cov-link]:  https://codecov.io/gh/phillyfan1138/GaussNewton


This is a basic implementation of the Gauss-Newton minimization for L2 norms.  This library requires my [Functional Utilities](https://github.com/phillyfan1138/FunctionalUtilities) and [AutoDiff](https://github.com/phillyfan1138/AutoDiff) libraries.

For verbose output, compile with flag `-D VERBOSE_FLAG=1`