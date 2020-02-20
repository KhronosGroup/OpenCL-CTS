//
// Copyright (c) 2017 The Khronos Group Inc.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#include "testBase.h"
#include "action_classes.h"


extern const char *IGetStatusString( cl_int status );

#define PRINT_OPS 0

int test_waitlist( cl_device_id device, cl_context context, cl_command_queue queue, Action *actionToTest, bool multiple )
{
    NDRangeKernelAction    actions[ 2 ];
    clEventWrapper events[ 3 ];
    cl_int status[ 3 ];
    cl_int error;

  if (multiple)
    log_info("\tExecuting reference event 0, then reference event 1 with reference event 0 in its waitlist, then test event 2 with reference events 0 and 1 in its waitlist.\n");
  else
    log_info("\tExecuting reference event 0, then test event 2 with reference event 0 in its waitlist.\n");

    // Set up the first base action to wait against
    error = actions[ 0 ].Setup( device, context, queue );
    test_error( error, "Unable to setup base event to wait against" );

    if( multiple )
    {
        // Set up a second event to wait against
        error = actions[ 1 ].Setup( device, context, queue );
        test_error( error, "Unable to setup second base event to wait against" );
    }

    // Now set up the actual action to test
    error = actionToTest->Setup( device, context, queue );
    test_error( error, "Unable to set up test event" );

    // Execute all events now
  if (PRINT_OPS) log_info("\tExecuting action 0...\n");
    error = actions[ 0 ].Execute( queue, 0, NULL, &events[ 0 ] );
    test_error( error, "Unable to execute first event" );

    if( multiple )
    {
    if (PRINT_OPS) log_info("\tExecuting action 1...\n");
        error = actions[ 1 ].Execute( queue, 1, &events[0], &events[ 1 ] );
        test_error( error, "Unable to execute second event" );
    }

    // Sanity check
  if( multiple ) {
    if (PRINT_OPS) log_info("\tChecking status of action 1...\n");
        error = clGetEventInfo( events[ 1 ], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof( status[ 1 ] ), &status[ 1 ], NULL );
    test_error( error, "Unable to get event status" );
  }
  if (PRINT_OPS) log_info("\tChecking status of action 0...\n");
    error = clGetEventInfo( events[ 0 ], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof( status[ 0 ] ), &status[ 0 ], NULL );
  test_error( error, "Unable to get event status" );

  log_info("\t\tEvent status after starting reference events: reference event 0: %s, reference event 1: %s, test event 2: %s.\n",
           IGetStatusString( status[ 0 ] ), (multiple ? IGetStatusString( status[ 1 ] ) : "N/A"), "N/A");

    if( ( status[ 0 ] == CL_COMPLETE ) || ( multiple && status[ 1 ] == CL_COMPLETE ) )
    {
        log_info( "WARNING: Reference event(s) already completed before we could execute test event! Possible that the reference event blocked (implicitly passing)\n" );
        return 0;
    }

  if (PRINT_OPS) log_info("\tExecuting action to test...\n");
    error = actionToTest->Execute( queue, ( multiple ) ? 2 : 1, &events[ 0 ], &events[ 2 ] );
    test_error( error, "Unable to execute test event" );

    // Hopefully, the first event is still running
  if (PRINT_OPS) log_info("\tChecking status of action to test 2...\n");
    error = clGetEventInfo( events[ 2 ], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof( status[ 2 ] ), &status[ 2 ], NULL );
    test_error( error, "Unable to get event status" );
  if( multiple ) {
    if (PRINT_OPS) log_info("\tChecking status of action 1...\n");
        error = clGetEventInfo( events[ 1 ], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof( status[ 1 ] ), &status[ 1 ], NULL );
    test_error( error, "Unable to get event status" );
  }
  if (PRINT_OPS) log_info("\tChecking status of action 0...\n");
    error = clGetEventInfo( events[ 0 ], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof( status[ 0 ] ), &status[ 0 ], NULL );
  test_error( error, "Unable to get event status" );

  log_info("\t\tEvent status after starting test event: reference event 0: %s, reference event 1: %s, test event 2: %s.\n",
           IGetStatusString( status[ 0 ] ), (multiple ? IGetStatusString( status[ 1 ] ) : "N/A"), IGetStatusString( status[ 2 ] ));

    if( multiple )
    {
        if( status[ 0 ] == CL_COMPLETE && status[ 1 ] == CL_COMPLETE )
        {
            log_info( "WARNING: Both events completed, so unable to test further (implicitly passing).\n" );
            clFinish( queue );
            return 0;
        }

    if(status[1] == CL_COMPLETE && status[0] != CL_COMPLETE)
    {
      log_error("ERROR: Test failed because the second wait event is complete and the first is not.(status: 0: %s and 1: %s)\n", IGetStatusString( status[ 0 ] ), IGetStatusString( status[ 1 ] ) );
            clFinish( queue );
            return -1;
    }
    }
    else
    {
        if( status[ 0 ] == CL_COMPLETE )
        {
            log_info( "WARNING: Reference event completed, so unable to test further (implicitly passing).\n" );
            clFinish( queue );
            return 0;
        }
        if( status[ 0 ] != CL_RUNNING && status[ 0 ] != CL_QUEUED && status[ 0 ] != CL_SUBMITTED )
        {
            log_error( "ERROR: Test failed because first wait event is not currently running, queued, or submitted! (status: 0: %s)\n", IGetStatusString( status[ 0 ] ) );
            clFinish( queue );
            return -1;
        }
    }

    if( status[ 2 ] != CL_QUEUED && status[ 2 ] != CL_SUBMITTED )
    {
        log_error( "ERROR: Test event is not waiting to run! (status: 2: %s)\n", IGetStatusString( status[ 2 ] ) );
        clFinish( queue );
        return -1;
    }

    // Now wait for the first reference event
  if (PRINT_OPS) log_info("\tWaiting for action 1 to finish...\n");
    error = clWaitForEvents( 1, &events[ 0 ] );
    test_error( error, "Unable to wait for reference event" );

    // Grab statuses again
  if (PRINT_OPS) log_info("\tChecking status of action to test 2...\n");
    error = clGetEventInfo( events[ 2 ], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof( status[ 2 ] ), &status[ 2 ], NULL );
    test_error( error, "Unable to get event status" );
  if( multiple ) {
    if (PRINT_OPS) log_info("\tChecking status of action 1...\n");
        error = clGetEventInfo( events[ 1 ], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof( status[ 1 ] ), &status[ 1 ], NULL );
    test_error( error, "Unable to get event status" );
  }
  if (PRINT_OPS) log_info("\tChecking status of action 0...\n");
    error = clGetEventInfo( events[ 0 ], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof( status[ 0 ] ), &status[ 0 ], NULL );
  test_error( error, "Unable to get event status" );

  log_info("\t\tEvent status after waiting for reference event 0: reference event 0: %s, reference event 1: %s, test event 2: %s.\n",
           IGetStatusString( status[ 0 ] ), (multiple ? IGetStatusString( status[ 1 ] ) : "N/A"), IGetStatusString( status[ 2 ] ));

    // Sanity
    if( status[ 0 ] != CL_COMPLETE )
    {
        log_error( "ERROR: Waited for first event but it's not complete (status: 0: %s)\n", IGetStatusString( status[ 0 ] ) );
        clFinish( queue );
        return -1;
    }

    // If we're multiple, and the second event isn't complete, then our test event should still be queued
    if( multiple && status[ 1 ] != CL_COMPLETE )
    {
    if( status[ 1 ] == CL_RUNNING && status[ 2 ] == CL_RUNNING ) {
      log_error("ERROR: Test event and second event are both running.\n");
      clFinish( queue );
      return -1;
    }
        if( status[ 2 ] != CL_QUEUED && status[ 2 ] != CL_SUBMITTED )
        {
            log_error( "ERROR: Test event did not wait for second event before starting! (status of ref: 1: %s, of test: 2: %s)\n", IGetStatusString( status[ 1 ] ), IGetStatusString( status[ 2 ] ) );
            clFinish( queue );
            return -1;
        }

        // Now wait for second event to complete, too
    if (PRINT_OPS) log_info("\tWaiting for action 1 to finish...\n");
        error = clWaitForEvents( 1, &events[ 1 ] );
        test_error( error, "Unable to wait for second reference event" );

        // Grab statuses again
    if (PRINT_OPS) log_info("\tChecking status of action to test 2...\n");
    error = clGetEventInfo( events[ 2 ], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof( status[ 2 ] ), &status[ 2 ], NULL );
    test_error( error, "Unable to get event status" );
    if( multiple ) {
      if (PRINT_OPS) log_info("\tChecking status of action 1...\n");
      error = clGetEventInfo( events[ 1 ], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof( status[ 1 ] ), &status[ 1 ], NULL );
      test_error( error, "Unable to get event status" );
    }
    if (PRINT_OPS) log_info("\tChecking status of action 0...\n");
    error = clGetEventInfo( events[ 0 ], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof( status[ 0 ] ), &status[ 0 ], NULL );
    test_error( error, "Unable to get event status" );

    log_info("\t\tEvent status after waiting for reference event 1: reference event 0: %s, reference event 1: %s, test event 2: %s.\n",
             IGetStatusString( status[ 0 ] ), (multiple ? IGetStatusString( status[ 1 ] ) : "N/A"), IGetStatusString( status[ 2 ] ));

        // Sanity
        if( status[ 1 ] != CL_COMPLETE )
        {
            log_error( "ERROR: Waited for second reference event but it didn't complete (status: 1: %s)\n", IGetStatusString( status[ 1 ] ) );
            clFinish( queue );
            return -1;
        }
    }

    // At this point, the test event SHOULD be running, but if it completed, we consider it a pass
    if( status[ 2 ] == CL_COMPLETE )
    {
        log_info( "WARNING: Test event already completed. Assumed valid.\n" );
        clFinish( queue );
        return 0;
    }
    if( status[ 2 ] != CL_RUNNING && status[ 2 ] != CL_SUBMITTED && status[ 2 ] != CL_QUEUED)
    {
        log_error( "ERROR: Second event did not start running after reference event(s) completed! (status: 2: %s)\n", IGetStatusString( status[ 2 ] ) );
        clFinish( queue );
        return -1;
    }

    // Wait for the test event, then return
  if (PRINT_OPS) log_info("\tWaiting for action 2 to test to finish...\n");
    error = clWaitForEvents( 1, &events[ 2 ] );
    test_error( error, "Unable to wait for test event" );

  error |= clGetEventInfo( events[ 2 ], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof( status[ 2 ] ), &status[ 2 ], NULL );
  test_error( error, "Unable to get event status" );

  log_info("\t\tEvent status after waiting for test event: reference event 0: %s, reference event 1: %s, test event 2: %s.\n",
           IGetStatusString( status[ 0 ] ), (multiple ? IGetStatusString( status[ 1 ] ) : "N/A"), IGetStatusString( status[ 2 ] ));

  // Sanity
  if( status[ 2 ] != CL_COMPLETE )
  {
    log_error( "ERROR: Test event didn't complete (status: 2: %s)\n", IGetStatusString( status[ 2 ] ) );
    clFinish( queue );
    return -1;
  }

  clFinish(queue);
    return 0;
}

#define TEST_ACTION( name ) \
    {    \
        name##Action action;    \
        log_info( "-- Testing " #name " (waiting on 1 event)...\n" );    \
        if( ( error = test_waitlist( deviceID, context, queue, &action, false ) ) != CL_SUCCESS )    \
            retVal++;            \
        clFinish( queue ); \
    }    \
    if( error == CL_SUCCESS )    /* Only run multiples test if single test passed */ \
    {    \
        name##Action action;    \
        log_info( "-- Testing " #name " (waiting on 2 events)...\n" );    \
        if( ( error = test_waitlist( deviceID, context, queue, &action, true ) ) != CL_SUCCESS )    \
            retVal++;            \
        clFinish( queue ); \
    }

int test_waitlists( cl_device_id deviceID, cl_context context, cl_command_queue oldQueue, int num_elements )
{
    cl_int error;
    int retVal = 0;
    cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;

    if( !checkDeviceForQueueSupport( deviceID, props ) )
    {
        log_info( "WARNING: Device does not support out-of-order exec mode; skipping test.\n" );
        return 0;
    }

    clCommandQueueWrapper queue = clCreateCommandQueue( context, deviceID, props, &error );
    test_error(error, "Unable to create out-of-order queue");

    log_info( "\n" );

    TEST_ACTION( NDRangeKernel )

    TEST_ACTION( ReadBuffer )
    TEST_ACTION( WriteBuffer )
    TEST_ACTION( MapBuffer )
    TEST_ACTION( UnmapBuffer )

    if( checkForImageSupport( deviceID ) == CL_IMAGE_FORMAT_NOT_SUPPORTED )
    {
        log_info( "\nNote: device does not support images. Skipping remainder of waitlist tests...\n" );
    }
    else
    {
        TEST_ACTION( ReadImage2D )
        TEST_ACTION( WriteImage2D )
        TEST_ACTION( CopyImage2Dto2D )
        TEST_ACTION( Copy2DImageToBuffer )
        TEST_ACTION( CopyBufferTo2DImage )
        TEST_ACTION( MapImage )

        if( checkFor3DImageSupport( deviceID ) == CL_IMAGE_FORMAT_NOT_SUPPORTED )
            log_info("Device does not support 3D images. Skipping remainder of waitlist tests...\n");
        else
        {
            TEST_ACTION( ReadImage3D )
            TEST_ACTION( WriteImage3D )
            TEST_ACTION( CopyImage2Dto3D )
            TEST_ACTION( CopyImage3Dto2D )
            TEST_ACTION( CopyImage3Dto3D )
            TEST_ACTION( Copy3DImageToBuffer )
            TEST_ACTION( CopyBufferTo3DImage )
        }
    }

    return retVal;
}

