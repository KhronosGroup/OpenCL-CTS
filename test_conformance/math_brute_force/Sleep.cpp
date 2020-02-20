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
#include "Sleep.h"
#include "Utility.h"

#if defined( __APPLE__ )
    #include <IOKit/pwr_mgt/IOPMLib.h>
    #include <IOKit/IOMessage.h>

    struct
    {
        io_connect_t            connection;
        IONotificationPortRef    port;
        io_object_t                iterator;
    }sleepInfo;

    void sleepCallback(    void *            refcon,
                        io_service_t        service,
                        natural_t        messageType,
                        void *            messageArgument );

    void sleepCallback(    void *            refcon UNUSED,
                        io_service_t        service UNUSED,
                        natural_t        messageType,
                        void *            messageArgument )
    {

        IOReturn result;
    /*
    service -- The IOService whose state has changed.
    messageType -- A messageType enum, defined by IOKit/IOMessage.h or by the IOService's family.
    messageArgument -- An argument for the message, dependent on the messageType.
    */
        switch ( messageType )
        {
            case kIOMessageSystemWillSleep:
                // Handle demand sleep (such as sleep caused by running out of
                // batteries, closing the lid of a laptop, or selecting
                // sleep from the Apple menu.
                IOAllowPowerChange(sleepInfo.connection,(long)messageArgument);
                vlog( "Hard sleep occurred.\n" );
                break;
            case kIOMessageCanSystemSleep:
                // In this case, the computer has been idle for several minutes
                // and will sleep soon so you must either allow or cancel
                // this notification. Important: if you don’t respond, there will
                // be a 30-second timeout before the computer sleeps.
                // IOCancelPowerChange(root_port,(long)messageArgument);
                result = IOCancelPowerChange(sleepInfo.connection,(long)messageArgument);
                if( kIOReturnSuccess != result )
                    vlog( "sleep prevention failed. (%d)\n", result);
            break;
            case kIOMessageSystemHasPoweredOn:
                // Handle wakeup.
                break;
        }
    }
#endif





void PreventSleep( void )
{
#if defined( __APPLE__ )
    vlog( "Disabling sleep... " );
    sleepInfo.iterator = (io_object_t) 0;
    sleepInfo.port = NULL;
    sleepInfo.connection = IORegisterForSystemPower
                            (
                                &sleepInfo,                    //void * refcon,
                                &sleepInfo.port,            //IONotificationPortRef * thePortRef,
                                sleepCallback,                //IOServiceInterestCallback callback,
                                &sleepInfo.iterator            //io_object_t * notifier
                            );

    if( (io_connect_t) 0 == sleepInfo.connection )
        vlog( "failed.\n" );
    else
        vlog( "done.\n" );

    CFRunLoopAddSource(CFRunLoopGetCurrent(),
                        IONotificationPortGetRunLoopSource(sleepInfo.port),
                        kCFRunLoopDefaultMode);
#else
    vlog( "*** PreventSleep() is not implemented on this platform.\n" );
#endif
}

void ResumeSleep( void )
{
#if defined( __APPLE__ )
    IOReturn result = IODeregisterForSystemPower ( &sleepInfo.iterator );
    if( 0 != result )
        vlog( "Got error %d restoring sleep \n", result );
    else
        vlog( "Sleep restored.\n" );
#else
    vlog( "*** ResumeSleep() is not implemented on this platform.\n" );
#endif
}



