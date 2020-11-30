// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean sameOrg=false)
{
    project.paths.construct_build_prefix()

    def getDependenciesCommand = ""
    if (project.installLibraryDependenciesFromCI)
    {
        project.libraryDependencies.each
        { libraryName ->
            getDependenciesCommand += auxiliary.getLibrary(libraryName, platform.jenkinsLabel, 'develop', sameOrg)
        }
    }

    String centos = platform.jenkinsLabel.contains('centos') ? 'source scl_source enable devtoolset-7' : ':'

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                ${getDependenciesCommand}
                ${centos}
                LD_LIBRARY_PATH=/opt/rocm/lib ${project.paths.build_command}
                """
    platform.runCommand(this, command)
}

def runTestCommand (platform, project)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release/clients/staging
                    ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./hipblas-test --gtest_output=xml --gtest_color=yes
                """

    platform.runCommand(this, command)
    junit "${project.paths.project_build_prefix}/build/release/clients/staging/*.xml"
}

def runPackageCommand(platform, project, jobName, label='')
{
    def command

    label = label != '' ? '-' + label.toLowerCase() : ''
    String ext = platform.jenkinsLabel.contains('ubuntu') ? "deb" : "rpm"
    String dir = jobName.contains('Debug') ? "debug" : "release"

    command = """
            set -x
            cd ${project.paths.project_build_prefix}/build/${dir}
            make package
            mkdir -p package
            if [ ! -z "$label" ]
            then
                for f in hipblas*.$ext
                do
                    mv "\$f" "hipblas${label}-\${f#*-}"
                done
            fi
            mv *.${ext} package/
        """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/${dir}/package/*.${ext}""")
}

return this
