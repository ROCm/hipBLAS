#!/usr/bin/env groovy

// Generated from snippet generator 'properties; set job properties'
properties([buildDiscarder(logRotator(
    artifactDaysToKeepStr: '',
    artifactNumToKeepStr: '',
    daysToKeepStr: '',
    numToKeepStr: '10')),
    disableConcurrentBuilds(),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])

////////////////////////////////////////////////////////////////////////
// Construct the path of the build directory
// This doesn't parse in a jenkinsfile because of returning multiple values
// https://issues.jenkins-ci.org/browse/JENKINS-38846
// def build_directory( String build_config, String root_path )
// {
//   String build_dir_rel = ""

//   if( build_config.equalsIgnoreCase( 'release' ) )
//   {
//     build_dir_rel = "build/release"
//   }
//   else
//   {
//     build_dir_rel = "build/debug"
//   }
//   String build_dir_abs = "${root_path}/${build_dir_rel}"

//   return [ build_dir_rel, build_dir_abs ]
// }

////////////////////////////////////////////////////////////////////////
// Construct the path of the build directory
def build_directory( String build_config, String root_path, StringBuilder build_dir_rel, StringBuilder build_dir_abs )
{
  if( build_config.equalsIgnoreCase( 'release' ) )
  {
    build_dir_rel = "build/release"
  }
  else
  {
    build_dir_rel = "build/debug"
  }
  build_dir_abs = "${root_path}/${build_dir_rel}"

  return void
}

////////////////////////////////////////////////////////////////////////
// Construct the path of the build directory
String build_directory_rel( String build_config )
{
  if( build_config.equalsIgnoreCase( 'release' ) )
  {
    return "build/release"
  }
  else
  {
    return "build/debug"
  }
}

String build_directory_abs( String build_config, String root_path )
{
    return "${root_path}/" + build_directory_rel( build_config )
}

////////////////////////////////////////////////////////////////////////
// This encapsulates running of unit tests
def docker_build_image(  )
{
  String project = "hipblas"
  String build_type_name = "build-ubuntu-16.04"
  String dockerfile_name = "dockerfile-${build_type_name}"
  String build_image_name = "${build_type_name}"
  def build_image = null

  stage('ubuntu-16.04 image')
  {
    dir('docker')
    {
      build_image = docker.build( "${project}/${build_image_name}:latest", "-f ${dockerfile_name} ." )
    }
  }

  return build_image
}

////////////////////////////////////////////////////////////////////////
// This encapsulates the cmake configure, build and package commands
// Leverages docker containers to encapsulate the build in a fixed environment
def docker_build_inside_image( def build_image, String build_config, String workspace_dir_abs )
{
  // def ( String build_dir_rel, String build_dir_abs ) = build_directory( build_config, workspace_dir_abs )
  String build_dir_rel = build_directory_rel( build_config );
  String build_dir_abs = build_directory_abs( build_config, workspace_dir_abs )

  // JENKINS-33510: the jenkinsfile dir() command is not workin well with docker.inside()
  build_image.inside( )
  {
    stage("build ${build_config}")
    {
      String install_prefix = "/opt/rocm"

      // Copy our rocBLAS dependency
        step ([$class: 'CopyArtifact',
                projectName: 'kknox/rocBLAS/copy-artifact']);

      // cmake -B${build_dir_abs} specifies to cmake where to generate build files
      // This is necessary because cmake seemingly randomly generates build makefile into the docker
      // workspace instead of the current set directory.  Not sure why, but it seems like a bug
      sh  """
          sudo dpkg -i library-build/*.deb
          mkdir -p ${build_dir_rel}
          cd ${build_dir_rel}
          cmake -B${build_dir_abs} \
            -DCMAKE_INSTALL_PREFIX=${install_prefix} \
            -DCMAKE_BUILD_TYPE=${build_config} \
            -DBUILD_LIBRARY=ON \
            -DBUILD_SHARED_LIBS=ON \
            -DCMAKE_PREFIX_PATH=/opt/rocm \
            ../..
          make -j\$(nproc)
        """
    }

    stage("packaging")
    {
      sh "cd ${build_dir_abs}; make package"
      archiveArtifacts artifacts: "${build_dir_rel}/*.deb", fingerprint: true
    }
  }

  return void
}

////////////////////////////////////////////////////////////////////////
// This encapsulates running of unit tests
def docker_upload_artifactory( String build_config, String workspace_dir_abs )
{
  def rocblas_install_image = null
  String image_name = "hipblas-ubuntu-16.04"
  String artifactory_org = "${env.JOB_NAME}".toLowerCase( )

  // def ( String build_dir_rel, String build_dir_abs ) = build_directory( build_config, workspace_dir_abs )
  StringBuilder build_dir_rel, build_dir_abs;
  build_directory( build_config, workspace_dir_abs, build_dir_rel, build_dir_abs )

  stage( 'artifactory' )
  {
    dir( "${build_dir_abs}/docker" )
    {
      //  We copy the docker files into the bin directory where the .deb lives so that it's a clean
      //  build everytime
      sh "cp -r ${workspace_dir_abs}/docker/* .; cp ${build_dir_abs}/*.deb ."
      rocblas_install_image = docker.build( "${artifactory_org}/${image_name}:${env.BUILD_NUMBER}", "-f dockerfile-${image_name} ." )
    }

    docker.withRegistry('http://compute-artifactory:5001', 'artifactory-cred' )
    {
      rocblas_install_image.push( "${env.BUILD_NUMBER}" )
      rocblas_install_image.push( 'latest' )
    }

    // Lots of images with tags are created above; no apparent way to delete images:tags with docker global variable
    // run bash script to clean images:tags after successful pushing
    sh "docker images | grep \"${artifactory_org}/${image_name}\" | awk '{print \$1 \":\" \$2}' | xargs docker rmi"
  }
}

////////////////////////////////////////////////////////////////////////
// This routines defines the build flow of the project
// Calls helper routines to do the work and stitches them together
def hipblas_build_pipeline( String build_type )
{
  // Convenience variables for common paths used in building
  String workspace_dir_abs = pwd()

  dir("${workspace_dir_abs}")
  {
    stage("github clone") {
      checkout scm
    }
  }

  // Create/reuse a docker image that represents the hipblas build environment
  def hipblas_build_image = docker_build_image( )

  // Build hipblas inside of the build environment
  docker_build_inside_image( hipblas_build_image, "${build_type}", "${workspace_dir_abs}" )

  // docker_upload_artifactory( "${build_type}", "${workspace_dir_abs}" )

  return void
}

////////////////////////////////////////////////////////////////////////
// This following are the nodes to build on
// Acts like a main() to the Jenkinsfile
//node('rocmtest')
node('docker && rocm')
{
  hipblas_build_pipeline( "Release" )
}
