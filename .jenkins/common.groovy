// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName)
{
    project.paths.construct_build_prefix()

    def getDependenciesCommand = ""
    if (project.installLibraryDependenciesFromCI)
    {
        project.libraryDependencies.each
        { libraryName ->
            getDependenciesCommand += auxiliary.getLibrary(libraryName, platform.jenkinsLabel, 'develop')
        }
    }
        
    if(jobName.contains('hipclang'))
    {
        command = """#!/usr/bin/env bash
                set -x
                ${getDependenciesCommand}
                cd ${project.paths.project_build_prefix}
                LD_LIBRARY_PATH=/opt/rocm/lib CXX=g++ ${project.paths.build_command} --hip-clang
                """
    }
    else
    {
        command = """#!/usr/bin/env bash
                set -x
                ${getDependenciesCommand}
                cd ${project.paths.project_build_prefix}
                LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=g++ ${project.paths.build_command}
                """
    }
    platform.runCommand(this, command)
}

def runTestCommand (platform, project)
{            
    def command

    if(platform.jenkinsLabel.contains('centos'))
    {
        command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}/build/release/clients/staging
                LD_LIBRARY_PATH=/opt/rocm/hcc/lib sudo ./example-sscal
                LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG sudo ./hipblas-test --gtest_output=xml --gtest_color=yes
                """
    }
    else
    {
        command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}/build/release/clients/staging
                LD_LIBRARY_PATH=/opt/rocm/hcc/lib ./example-sscal
                LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./hipblas-test --gtest_output=xml --gtest_color=yes
                """
    }

    platform.runCommand(this, command)
    junit "${project.paths.project_build_prefix}/build/release/clients/staging/*.xml"
}

def runPackageCommand(platform, project, jobName)
{
    def command 

    if(platform.jenkinsLabel.contains('centos'))
    {
        command = """
                set -x
                cd ${project.paths.project_build_prefix}/build/release
                make package
                rm -rf package && mkdir -p package
                mv *.rpm package/
                rpm -qlp package/*.rpm
            """

        platform.runCommand(this, command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.rpm""")        
    }
    else
    {
        command = """
                set -x
                cd ${project.paths.project_build_prefix}/build/release
                make package
                rm -rf package && mkdir -p package
                mv *.deb package/
                dpkg -c package/*.deb
                """

        platform.runCommand(this, command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.deb""")
    }
}

return this
