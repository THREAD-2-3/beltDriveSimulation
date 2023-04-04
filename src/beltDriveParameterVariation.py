#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Model for belt drive; special case according to: A. Pechstein, J. Gerstmayr. A Lagrange-Eulerian formulation of an axially moving beam based on the absolute nodal coordinate formulation, Multibody System Dynamics, Vol. 30 (3), 343 – 358, 2013.
#
# Author:   Johannes Gerstmayr, Konstantina Ntarladima
# Date:     2022-02-27
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Section 1: Import necessary modules 
import exudyn as exu
from exudyn.itemInterface import *
from exudyn.utilities import *
from exudyn.beams import *
from exudyn.plot import DataArrayFromSensorList

import numpy as np
from math import sin, cos, pi, sqrt , asin, acos, atan2, exp
import copy 
#%%
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Section 2 
SC = exu.SystemContainer()  # contains the systems (most important), 
                            # solvers (static, dynamics, ...), 
                            # visualization settings
mbs = SC.AddSystem()  # contains everything that defines a solvable multibody 
                      # system; a large set of nodes, objects, markers,
                      # loads can added to the system

fileDir = 'solution_nosync/' # folder name for the solutions


#%%
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Section 3: Parameter function
#this is the function which is repeatedly called from ParameterVariation
#parameterSet contains dictionary with varied parameters
def ParameterFunction(parameterSet):
    mbs.Reset() # clears the system and allows
                # to set up a new system
#%%
    #++++++++++++++++++++++++++++++++++++++++++++++
    # Subsection 3.A
    #++++++++++++++++++++++++++++++++++++++++++++++
    #store default parameters in structure (all these parameters can be varied!)
    class P: pass   # create empty structure for parameters; 
                    # simplifies way to update parameters

    #default values
    #P.computationIndex = 'Ref'
    P.tEnd = 1                          #2.45 used 
    P.stepSize = 5e-5
    P.dryFriction = 0.5
    P.contactStiffnessPerArea = 1e8         # N/m^2
    P.frictionStiffnessPerArea = 1e8*10*5   # N/m^2, then re-computed per segment
    P.nSegments = 4                         # 4, for nANCFnodes=60, P.nSegments = 2 
                                            # does not converge for static solution
    P.nANCFnodes = 2*60                     # 120 works well, 60 leads to oscillatory 
                                            # tangent/normal forces 
    
    # Now update parameters with parameterSet (will work with any parameters in 
    # structure P)
    for key,value in parameterSet.items():
        setattr(P,key,value)
#%%
    #++++++++++++++++++++++++++++++++++++++++++++++
    # Subsection 3.B 
    #++++++++++++++++++++++++++++++++++++++++++++++
    # START HERE: create parameterized model, using structure P, which is updated 
    # in every computation

    useGraphics = False
    runParallel = False
    if 'functionData' in parameterSet:
        functionData = parameterSet['functionData']
        if 'useGraphics' in functionData:
            useGraphics = functionData['useGraphics']
        if 'computationIndex' in functionData: #this is added in parallel case
            runParallel = True
 
    #%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Parameters for the belt
    gVec = [0,-9.81*1,0]     # gravity
    Emodulus=1e7             # Young's modulus of ANCF element in N/m^2
    b=0.08 #0.002            # width of rectangular ANCF element in m
    hc = 0.01*0.01           # height (geometric) of rectangular ANCF element in m
    hcStiff = 0.01           # stiffness relevant height
    rhoBeam=1036.            # density of ANCF element in kg/m^3
    A=b*hcStiff              # cross sectional area of ANCF element in m^2
    I=(b*hcStiff**3)/12      # second moment of area of ANCF element in m^4
    EI = Emodulus*I*0.02     # bending stiffness
    EA = Emodulus*A          # axial stiffness
    rhoA = rhoBeam*A
    dEI = 0*1e-3*Emodulus*I         # REMARK: bending proportional damping. 
                                    # Set zero in the 2013 paper there is not. 
                                    # We need the damping for changing the initial 
                                    # configuration.
    #dEA = 0.1*1e-2*Emodulus*A      # axial strain proportional damping. Same 
                                    # as for the bending damping.
    dEA = 1                         # dEA=1 in paper PechsteinGerstmayr 2013, 
                                    # according to HOTINT C++ files ...   
    #%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Settings:
    useContact = True
    doDynamic = True
    makeAnimation = False
    velocityControl = True
    staticEqulibrium = True        
    useBristleModel = True
    
    preCurved = False               # uses preCurvature according to straight 
                                    # and curved initial segments
                                    # In 2013 paper, reference curvature is set 
                                    # according to initial geometry and released 
                                    # until tAccStart
    strainIsRelativeToReference = False# 0: straight reference, 1.: curved reference
    
    useContactCircleFriction = True
    
    movePulley = False              # True for 2013 paper, move within first 0.05 
                                    # seconds; but this does not work with 
                                    # Index 2 solver
    
    discontinuousIterations = 5     # larger is more accurate, but smaller step size
                                    # is equivalent
    #%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Parameters for dynamic simulation     
    tAccStart = 0.05
    tAccEnd = 0.6
    omegaFinal = 12
    tTorqueStart = 1.
    tTorqueEnd = 1.5
    #%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Parameters for the contact     
    useFriction = True
    contactStiffness = P.contactStiffnessPerArea*40
    contactDamping = 0
    #%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Parameters for the wheels
    wheelMass = 50                  # 1 the wheel mass is not given in the paper, 
                                    # only the inertia 
    # for the second wheel 
    wheelInertia = 0.25             # 0.01
    rotationDampingWheels = 2       # zero in example in 2013 paper; torque 
                                    # proportional to rotation speed
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Pulleys' locations
    # For complicated shape:
    
    initialDisplacement0 = 0
    radiusPulley = 0.09995
    distancePulleys = 0.1*pi
    initialDistance = positionPulley2x 
    preStretch = -0.05
          
    #%%
    # Add belt
    # Create geometry:
    circleList = [[[initialDisplacement0,0], radiusPulley,'L'],
                  [[distancePulleys,0], radiusPulley,'L'],]
    
    reevingDict = CreateReevingCurve(circleList, drawingLinesPerCircle = 64, 
                                    radialOffset=0.5*hc, closedCurve=True, 
                                    numberOfANCFnodes=P.nANCFnodes, graphicsNodeSize= 0.01)
    
    # Set precurvature at location of pulleys:
    elementCurvatures = [] #no pre-curvatures
    if preCurved:
        elementCurvatures = reevingDict['elementCurvatures']
    
    gList=[]

    oGround=mbs.AddObject(ObjectGround(referencePosition= [0,0,0], visualization=VObjectGround(show=False)))
    nGround = mbs.AddNode(NodePointGround())
    mCoordinateGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGround, coordinate=0))
    
    #create ANCF elements:
    dimZ = b #z.dimension
    
    cableTemplate = Cable2D(
                            physicsMassPerLength = rhoA,
                            physicsBendingStiffness = EI,
                            physicsAxialStiffness = EA,
                            physicsBendingDamping = dEI,
                            physicsAxialDamping = dEA,
                            physicsReferenceAxialStrain = preStretch, 
                            physicsReferenceCurvature = 0.,
                            useReducedOrderIntegration = 2, 
                            strainIsRelativeToReference = strainIsRelativeToReference,
                            visualization=VCable2D(drawHeight=hc),
                            )
    
    ancf = PointsAndSlopes2ANCFCable2D(mbs, reevingDict['ancfPointsSlopes'], 
                                       reevingDict['elementLengths'], 
                                       cableTemplate, massProportionalLoad=gVec, 
                                       fixedConstraintsNode0=[1*staticEqulibrium,0,0,0],
                                       elementCurvatures  = elementCurvatures,
                                       firstNodeIsLastNode=True, graphicsSizeConstraints=0.01)
    
    if useContactCircleFriction:
        lElem = reevingDict['totalLength'] / P.nANCFnodes
        cFact=b*lElem/P.nSegments                 # stiffness shall be per area,
                                                  # but is applied at every segment
        if not runParallel: print('cFact=',cFact, ', lElem=', lElem)
    
        contactStiffness*=cFact
        contactDamping = 2000*cFact*40     # according to Dufva 2008 paper ... 
                                           # seems also to be used in 2013 
                                           # Pechstein Gerstmayr
        if useBristleModel:
            frictionStiffness = P.frictionStiffnessPerArea*cFact      #1e7 converges good; 
                                                                      #1e8 is already 
                                                                      #quite accurate
             
            massSegment = rhoA*lElem/P.nSegments
            frictionVelocityPenalty = sqrt(frictionStiffness*massSegment)*10 #bristle damping; 
                           #should be adjusted to reduce vibrations induced by bristle model
        else:
            frictionVelocityPenalty = 0.1*1e7*cFact # 1e7 is original in 2013 paper; 
                                                    # requires smaller time step
            frictionStiffness = 0                   # as in 2013 paper
    

    if not runParallel: 
        print('contactStiffness=',contactStiffness/cFact,' per area')
        print('contactDamping=',contactDamping/cFact,' per area')
        print('frictionStiffness=',frictionStiffness/cFact,' per area')
        print('frictionVelocityPenalty=',frictionVelocityPenalty/cFact,' per area')
        print('EA=',EA)
        print('EI=',EI)
        print('beam height=',hc)


    #++++++++++++++++++++++++++++++++++++++++++++++
    # Subsection 3.C: adding sensors
    #++++++++++++++++++++++++++++++++++++++++++++++
        
    #create sensors for all nodes
    sMidVel = []
    sAxialForce = []
    sCable0Pos = []
    
    ancfNodes = ancf[0]
    ancfObjects = ancf[1]
    positionList2Node = []               #axial position at x=0 and x=0.5*lElem
    positionListMid = []                 #axial position at midpoint of element
    positionListSegments = []            #axial position at midpoint of segments
    currentPosition = 0                  #is increased at every iteration
    for i,obj in enumerate(ancfObjects):
        lElem = reevingDict['elementLengths'][i]
        positionList2Node += [currentPosition, currentPosition + 0.5*lElem]
        positionListMid += [currentPosition + 0.5*lElem]
    
        for j in range(P.nSegments):
            segPos = (j+0.5)*lElem/P.nSegments + currentPosition
            positionListSegments += [segPos]
        currentPosition += lElem
    
        sAxialForce += [mbs.AddSensor(SensorBody(bodyNumber = obj, 
                                                  storeInternal=True,
                                                  localPosition=[0.*lElem,0,0], 
                                                  outputVariableType=exu.OutputVariableType.ForceLocal))]
        sAxialForce += [mbs.AddSensor(SensorBody(bodyNumber = obj, 
                                                  storeInternal=True,
                                                  localPosition=[0.5*lElem,0,0], 
                                                  outputVariableType=exu.OutputVariableType.ForceLocal))]
        sMidVel += [mbs.AddSensor(SensorBody(bodyNumber = obj, 
                                              storeInternal=True,
                                              localPosition=[0.5*lElem,0,0], #0=at left node
                                              outputVariableType=exu.OutputVariableType.VelocityLocal))]
        sCable0Pos += [mbs.AddSensor(SensorBody(bodyNumber = obj, 
                                                storeInternal=True,
                                                localPosition=[0.*lElem,0,0],
                                                outputVariableType=exu.OutputVariableType.Position))]
    
        
    fileClassifier  = ''
    fileClassifier += '-tt'+str(int(P.tEnd*100))
    fileClassifier += '-hh'+str(int(P.stepSize/1e-6))
    fileClassifier += '-nn'+str(int(P.nANCFnodes/60))
    fileClassifier += '-ns'+str(P.nSegments)
    fileClassifier += '-cs'+str(int((P.contactStiffnessPerArea/1e7)))
    fileClassifier += '-fs'+str(int((P.frictionStiffnessPerArea/1e7)))
    fileClassifier += '-df'+str(int(P.dryFriction*10))
    fileClassifier += '-'
        
    #%%
    #++++++++++++++++++++++++++++++++++++++++++++++
    # Subsection 3.D: adding contact
    #++++++++++++++++++++++++++++++++++++++++++++++

    if useContact:
    
        contactObjects = [[],[]] #list of contact objects
        
    
        dimZ= 0.01 #for drawing
        sWheelRot = [] #sensors for angular velocity
    
        nMassList = []
        wheelSprings = [] #for static computation
        for i, wheel in enumerate(circleList):
            p = [wheel[0][0], wheel[0][1], 0]         #position of wheel center
            r = wheel[1]
        
            rot0 = 0                                  #initial rotation
            pRef = [p[0], p[1], rot0]
            gList = [GraphicsDataCylinder(pAxis=[0,0,-dimZ],vAxis=[0,0,-dimZ], radius=r,
                                          color= color4dodgerblue, nTiles=64),
                     GraphicsDataArrow(pAxis=[0,0,0], vAxis=[-0.9*r,0,0], radius=0.01*r, color=color4orange),
                     GraphicsDataArrow(pAxis=[0,0,0], vAxis=[0.9*r,0,0], radius=0.01*r, color=color4orange)]
    
            omega0 = 0                                #initial angular velocity
            v0 = np.array([0,0,omega0]) 
    
            nMass = mbs.AddNode(NodeRigidBody2D(referenceCoordinates=pRef, initialVelocities=v0,
                                                visualization=VNodeRigidBody2D(drawSize=dimZ*2)))
            nMassList += [nMass]
            oMass = mbs.AddObject(ObjectRigidBody2D(physicsMass=wheelMass, physicsInertia=wheelInertia,
                                                    nodeNumber=nMass, visualization=
                                                    VObjectRigidBody2D(graphicsData=gList)))
            mNode = mbs.AddMarker(MarkerNodeRigid(nodeNumber=nMass))
            mGroundWheel = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition=p, visualization = VMarkerBodyRigid(show = False)))
        
            #mbs.AddObject(RevoluteJoint2D(markerNumbers=[mGroundWheel, mNode], visualization=VRevoluteJoint2D(show=False)))
    
            mCoordinateWheelX = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nMass, coordinate=0))
            mCoordinateWheelY = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nMass, coordinate=1))
            constraintX = mbs.AddObject(CoordinateConstraint(markerNumbers=[mCoordinateGround, mCoordinateWheelX],
                                                     visualization=VCoordinateConstraint(show = False)))
            constraintY = mbs.AddObject(CoordinateConstraint(markerNumbers=[mCoordinateGround, mCoordinateWheelY],
                                                     visualization=VCoordinateConstraint(show = False)))
            if i==0:
                constraintPulleyLeftX = constraintX
    
            if True:
            
                sWheelRot += [mbs.AddSensor(SensorNode(nodeNumber=nMass, 
                                                       storeInternal=True,
                                                       fileName=fileDir+'wheel'+str(i)+'angVel'+fileClassifier+'.txt',
                                                       outputVariableType=exu.OutputVariableType.AngularVelocity))]
            tdisplacement = 0.05
      
                             
            def UFvelocityDrive(mbs, t, itemNumber, lOffset): #time derivative of UFoffset
                if t < tAccStart:
                    v = 0
                if t >= tAccStart and t < tAccEnd:
                    v = omegaFinal/(tAccEnd-tAccStart)*(t-tAccStart)
                elif t >= tAccEnd:
                    v = omegaFinal
                return v    
            
            if doDynamic:    
                if i == 0:
                    if velocityControl:
                        mCoordinateWheel = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nMass, coordinate=2))
                        velControl = mbs.AddObject(CoordinateConstraint(markerNumbers=[mCoordinateGround, mCoordinateWheel],
                                                            velocityLevel=True, offsetUserFunction_t= UFvelocityDrive,
                                                            visualization=VCoordinateConstraint(show = False)))#UFvelocityDrive
                        sTorquePulley0 = mbs.AddSensor(SensorObject(objectNumber=velControl, 
                                                               fileName=fileDir+'torquePulley0'+fileClassifier+'.txt',
                                                               outputVariableType=exu.OutputVariableType.Force))
                if i == 1:
                    mCoordinateWheel = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nMass, coordinate=2))
                    mbs.AddObject(CoordinateSpringDamper(markerNumbers=[mCoordinateGround, mCoordinateWheel],
                                                         damping = rotationDampingWheels,
                                                         visualization=VCoordinateSpringDamper(show = False)))
                    
                    #this is used for times > 1 in order to see influence of torque step in Wheel1
                    def UFforce(mbs, t, load):
                        tau = 0.

                        tau +=  25.*(SmoothStep(t, tTorqueStart, tTorqueEnd, 0., 1.) - SmoothStep(t, 3.5, 4., 0., 1.))
                        return -tau
                    
                    loadPulley1 = mbs.AddLoad(LoadCoordinate(markerNumber=mCoordinateWheel,
                                               load = 0, loadUserFunction = UFforce))
                    sTorquePulley1 = mbs.AddSensor(SensorLoad(loadNumber=loadPulley1, 
                                                            fileName=fileDir+'torquePulley1'+fileClassifier+'.txt'
                                                            ))
    
            if staticEqulibrium:
                mCoordinateWheel = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nMass, coordinate=2))
                csd = mbs.AddObject(CoordinateConstraint(markerNumbers=[mCoordinateGround, mCoordinateWheel],
                                                         visualization=VCoordinateConstraint(show = False)))
                wheelSprings += [csd]
            
    
    
            cableList = ancf[1]
            mCircleBody = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oMass))
            for k in range(len(cableList)):
                initialGapList = [0.1]*P.nSegments + [-2]*(P.nSegments) + [0]*(P.nSegments) #initial gap of 0., isStick (0=slip, +-1=stick, -2 undefined initial state), lastStickingPosition (0)

                mCable = mbs.AddMarker(MarkerBodyCable2DShape(bodyNumber=cableList[k], 
                                                              numberOfSegments = P.nSegments, verticalOffset=-hc/2))
                nodeDataContactCable = mbs.AddNode(NodeGenericData(initialCoordinates=initialGapList,
                                                                   numberOfDataCoordinates=P.nSegments*(1+2) ))

                co = mbs.AddObject(ObjectContactFrictionCircleCable2D(markerNumbers=[mCircleBody, mCable], nodeNumber = nodeDataContactCable, 
                                                         numberOfContactSegments=P.nSegments, 
                                                         contactStiffness = contactStiffness, 
                                                         contactDamping=contactDamping, 
                                                         frictionVelocityPenalty = frictionVelocityPenalty, 
                                                         frictionStiffness = frictionStiffness, 
                                                         frictionCoefficient=int(useFriction)*P.dryFriction,
                                                         circleRadius = r,
                                                         # useSegmentNormals=False,
                                                         visualization=VObjectContactFrictionCircleCable2D(showContactCircle=False)))
                contactObjects[i] += [co]
    
    #++++++++++++++++++++++++++++++++++++++++++++++
    # Subsection 3.E: adding sensors for contact
    #++++++++++++++++++++++++++++++++++++++++++++++
    sContactDisp = [[],[]]
    sContactForce = [[],[]]
    for i in range(len(contactObjects)):
        for obj in contactObjects[i]:
            sContactForce[i] += [mbs.AddSensor(SensorObject(objectNumber = obj, 
                                                            storeInternal=True,
                                                            outputVariableType=exu.OutputVariableType.ForceLocal))]
            sContactDisp[i] += [mbs.AddSensor(SensorObject(objectNumber = obj, 
                                                            storeInternal=True,
                                                            outputVariableType=exu.OutputVariableType.Coordinates))]
    #++++++++++++++++++++++++++++++++++++++++++++++            
    #++++++++++++++++++++++++++++++++++++++++++++++       
            
    
    
    #user function to smoothly transform from curved to straight reference configuration as
    #in paper 2013, Pechstein, Gerstmayr
    def PreStepUserFunction(mbs, t):
    
        if True and t <= tAccStart+1e-10:
            cableList = ancf[1]
            fact = (tAccStart-t)/tAccStart #from 1 to 0
            if fact < 1e-12: fact = 0. #for very small values ...
            #curvatures = reevingDict['elementCurvatures']
            #print('fact=', fact)
            for i in range(len(cableList)):
                oANCF = cableList[i]
                mbs.SetObjectParameter(oANCF, 'strainIsRelativeToReference', 
                                       fact)
                mbs.SetObjectParameter(oANCF, 'physicsReferenceAxialStrain', 
                                        preStretch*(1.-fact))
    
        
        return True
    
    
    mbs.Assemble()
    
#%% 
#Section 4: Simulation settings

    simulationSettings = exu.SimulationSettings() #takes currently set values or default values
    
    simulationSettings.linearSolverType = exu.LinearSolverType.EigenSparse
    simulationSettings.solutionSettings.coordinatesSolutionFileName = fileDir+'coordinatesSolution.txt'
    
    simulationSettings.solutionSettings.writeSolutionToFile = False
    simulationSettings.solutionSettings.solutionWritePeriod = 0.002
    simulationSettings.solutionSettings.sensorsWritePeriod = 0.001
    simulationSettings.displayComputationTime = False #useGraphics
    simulationSettings.parallel.numberOfThreads = 1 #use 4 to speed up for > 100 ANCF elements
    simulationSettings.displayStatistics = False
    
    simulationSettings.timeIntegration.endTime = P.tEnd
    simulationSettings.timeIntegration.numberOfSteps = int(P.tEnd/P.stepSize)
    simulationSettings.timeIntegration.stepInformation= 255
    
    simulationSettings.timeIntegration.verboseMode = 1-int(runParallel)
    
    simulationSettings.timeIntegration.newton.useModifiedNewton = True
 
    simulationSettings.timeIntegration.discontinuous.iterationTolerance = 1e-3
    simulationSettings.timeIntegration.discontinuous.maxIterations = discontinuousIterations #3
    
    
    SC.visualizationSettings.general.circleTiling = 24
    SC.visualizationSettings.loads.show=False
    SC.visualizationSettings.sensors.show=False
    SC.visualizationSettings.markers.show=False
    SC.visualizationSettings.nodes.defaultSize = 0.002
    SC.visualizationSettings.openGL.multiSampling = 4
    SC.visualizationSettings.openGL.lineWidth = 2
    SC.visualizationSettings.window.renderWindowSize = [1920,1080]
    
    SC.visualizationSettings.connectors.showContact = True
    SC.visualizationSettings.contact.contactPointsDefaultSize = 0.0002
    SC.visualizationSettings.contact.showContactForces = True
    SC.visualizationSettings.contact.contactForcesFactor = 0.005
    
    if makeAnimation == True:
        simulationSettings.solutionSettings.recordImagesInterval = 0.02
        SC.visualizationSettings.exportImages.saveImageFileName = "animationNew/frame"
    
    
    if True:
        SC.visualizationSettings.bodies.beams.axialTiling = 4
        SC.visualizationSettings.bodies.beams.drawVertical = True
        SC.visualizationSettings.bodies.beams.drawVerticalLines = True
    
        SC.visualizationSettings.contour.outputVariableComponent=0
        SC.visualizationSettings.contour.outputVariable=exu.OutputVariableType.ForceLocal
        SC.visualizationSettings.bodies.beams.drawVerticalFactor = 0.0003
        SC.visualizationSettings.bodies.beams.drawVerticalOffset = -220
            
        SC.visualizationSettings.bodies.beams.reducedAxialInterploation = True

    if useGraphics: 
        exu.StartRenderer()
 
    simulationSettings.staticSolver.adaptiveStep = False
    simulationSettings.staticSolver.loadStepGeometric = True;
    simulationSettings.staticSolver.loadStepGeometricRange=1e4
    simulationSettings.staticSolver.numberOfLoadSteps = 10
    #simulationSettings.staticSolver.useLoadFactor = False
    simulationSettings.staticSolver.stabilizerODE2term = 1e5
    simulationSettings.staticSolver.newton.relativeTolerance = 1e-6
    simulationSettings.staticSolver.newton.absoluteTolerance = 1e-6

#%%
# Section 5: Performing static and dynamic simulation    
       
    if staticEqulibrium: #precompute static equilibrium
        mbs.SetObjectParameter(velControl, 'activeConnector', False)
    
        for i in range(len(contactObjects)):
            for obj in contactObjects[i]:
                mbs.SetObjectParameter(obj, 'frictionCoefficient', 0.)
                mbs.SetObjectParameter(obj, 'frictionStiffness', 1e-8) #do not 
                         #set to zero, as it needs to do some initialization...

        exu.SolveStatic(mbs, simulationSettings, updateInitialValues=True) 
    
        #check total force on support, expect: supportLeftX \approx 2*preStretch*EA
        supportLeftX = mbs.GetObjectOutput(constraintPulleyLeftX,variableType=exu.OutputVariableType.Force)
        if not runParallel:
            print('Force x in support of left pulley = ', supportLeftX)
            print('Belt pre-tension=', preStretch*EA)
        
        for i in range(len(contactObjects)):
            for obj in contactObjects[i]:
                mbs.SetObjectParameter(obj, 'frictionCoefficient', P.dryFriction)
                mbs.SetObjectParameter(obj, 'frictionStiffness', frictionStiffness)
    
        for coordinateConstraint in ancf[4]:
            mbs.SetObjectParameter(coordinateConstraint, 'activeConnector', False)
            
        mbs.SetObjectParameter(velControl, 'activeConnector', True)
        for csd in wheelSprings:
            mbs.SetObjectParameter(csd, 'activeConnector', False)
    else:
        mbs.SetPreStepUserFunction(PreStepUserFunction)
    
    exu.SolveDynamic(mbs, simulationSettings, solverType=exu.DynamicSolverType.TrapezoidalIndex2) #183 Newton iterations, 0.114 seconds

    
    if useGraphics and False:
        SC.visualizationSettings.general.autoFitScene = False
        SC.visualizationSettings.general.graphicsUpdateInterval=0.02
        from exudyn.interactive import SolutionViewer
        sol = LoadSolutionFile(fileDir+'coordinatesSolution.txt', safeMode=True)#, maxRows=100)
        SolutionViewer(mbs, sol)
    
    
    if useGraphics: 
        # SC.WaitForRenderEngineStopFlag()
        exu.StopRenderer() #safely close rendering window!

    #shift data depending on axial position by subtracting xOff; put negative x values+shiftValue to end of array
    def ShiftXoff(data, xOff, shiftValue):
        indOff = 0
        n = data.shape[0]
        data[:,0] -= xOff
        for i in range(n):
           if data[i,0] < 0:
               indOff+=1
               data[i,0] += shiftValue
        data = np.vstack((data[indOff:,:], data[0:indOff,:]))
        return data

#%%
# Section 6: Post process data           
        
    #compute axial offset, to normalize results:
    nodePos0 = mbs.GetSensorValues(sCable0Pos[0])
    xOff = nodePos0[0]

    
    #find closest node:
    minDist = 1e10
    iDist = -1
    p0 = np.array([0.,radiusPulley,0.])
    pDist = [] #stored position
    for (i,sPos) in enumerate(sCable0Pos):
        p = mbs.GetSensorValues(sPos)
        #print("pos ",i,'=',p)
        dist = np.linalg.norm(p0-p)
        if dist < minDist and p[0] >= 0:
            iDist = i
            minDist = dist
            pDist = p
    lElem = reevingDict['elementLengths'][0] #equidistant ...
    xOff = iDist * lElem - pDist[0]
    correctXoffset = True
    if not runParallel: print('iDist=',iDist, ', xOff=', xOff, 'pDist[0]=', pDist[0])
    
    dataVel = DataArrayFromSensorList(mbs, sensorNumbers=sMidVel, positionList=positionListMid)
    if correctXoffset:
        dataVel=ShiftXoff(dataVel,xOff, reevingDict['totalLength'])
    
    #axial force over beam length:
    dataForce = DataArrayFromSensorList(mbs, sensorNumbers=sAxialForce, positionList=positionList2Node)
    if correctXoffset:
        dataForce = ShiftXoff(dataForce,xOff, reevingDict['totalLength'])

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #contact forces are stored (x/y) for every segment ==> put into consecutive array
    contactForces =[[],[]] #these are the contact forces of the whole belt, but from both pulleys!
    for i in range(len(sContactForce)):
        contactForces[i] = np.zeros((len(sContactForce[i])*P.nSegments, 3)) #per row: [position, Fx, Fy]
        for j, sensor in enumerate(sContactForce[i]):
            values = mbs.GetSensorValues(sensor)
            for k in range(P.nSegments):
                row = j*P.nSegments + k
                contactForces[i][row,0] = positionListSegments[row]
                contactForces[i][row, 1:] = (1/cFact)*values[k*2:k*2+2] #convert into pressure

    contactForcesTotal = contactForces[0]
    contactForcesTotal[:,1:] += contactForces[1][:,1:]

    if correctXoffset:
        contactForcesTotal = ShiftXoff(contactForcesTotal,xOff, reevingDict['totalLength'])

    contactDisp =[[],[]] #slip and gap
    for i in range(len(sContactDisp)):
        contactDisp[i] = np.zeros((len(sContactDisp[i])*P.nSegments, 3)) #per row: [position, Fx, Fy]
        for j, sensor in enumerate(sContactDisp[i]):
            values = mbs.GetSensorValues(sensor)
            for k in range(P.nSegments):
                row = j*P.nSegments + k
                contactDisp[i][row,0] = positionListSegments[row]
                contactDisp[i][row, 1:] = values[k*2:k*2+2]

    contactDispTotal = contactDisp[0]
    contactDispTotal[:,1:] += contactDisp[1][:,1:]

    if correctXoffset:
        contactDispTotal = ShiftXoff(contactDispTotal,xOff, reevingDict['totalLength'])

    header  = 'Exudyn output file\n' #comment symbol is added to lines by np.savetxt !
    header += 'Exudyn version='+str(exu.__version__)+'\n'
    header += 'endTime='+str(P.tEnd)+'\n'
    header += 'stepSize='+str(P.stepSize)+'\n'
    header += 'nSegments='+str(P.nSegments)+'\n'
    header += 'nANCFnodes='+str(P.nANCFnodes)+'\n'
    header += 'contactStiffnessPerArea='+str(P.contactStiffnessPerArea)+'\n'
    header += 'frictionStiffnessPerArea='+str(P.frictionStiffnessPerArea)+'\n'
    header += 'contactStiffness='+str(contactStiffness)+'\n'
    header += 'frictionStiffness='+str(frictionStiffness)+'\n'
    header += 'contactDamping='+str(contactDamping)+'\n'
    header += 'frictionVelocityPenalty='+str(frictionVelocityPenalty)+'\n'
    header += 'dryFriction='+str(P.dryFriction)+'\n'

    #export solution:

    np.savetxt(fileDir+'beamVel'+fileClassifier+'.txt', dataVel, delimiter=',', 
               header='Exudyn: solution of belt drive, beam velocities over belt length\n'+header, encoding=None)
    np.savetxt(fileDir+'beamForce'+fileClassifier+'.txt', dataForce, delimiter=',', 
               header='Exudyn: solution of belt drive, beam (axial) forces over belt length\n'+header, encoding=None)
    np.savetxt(fileDir+'contactForces'+fileClassifier+'.txt', contactForcesTotal, delimiter=',', 
               header='Exudyn: solution of belt drive, contact forces over belt length\n'+header, encoding=None)
    np.savetxt(fileDir+'contactDisp'+fileClassifier+'.txt', contactDispTotal, delimiter=',', 
               header='Exudyn: solution of belt drive, slip and gap over belt length\n'+header, encoding=None)
    
    return fileClassifier



#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Section 7: Select single simulation or variation. 
#              Values for parameter variations are defined here.   
performVariation = True
performSingleSimulation = False
performPlots = True #and not performVariation ## CHECK!!

if __name__ == '__main__': #include this to enable parallel processing
    from exudyn.processing import ParameterVariation, ProcessParameterList
    import time
    if performVariation:
    
        functionData = {'useGraphics':False}
        runParallel = True
        f=1
        # f=0.002
        # takes 4001 seconds for 11 cases on I9
        parList = [
            #this is the reference solution:
            {'tEnd':2.45*f, 'nANCFnodes':480, 'stepSize':1e-5, 'functionData':functionData}, 
            #investigate different element numbers
            {'tEnd':2.45*f, 'nANCFnodes':60,'nSegments':4,'functionData':functionData}, #at least 4 segments necessary!
            {'tEnd':2.45*f, 'nANCFnodes':120,'functionData':functionData},
            {'tEnd':2.45*f, 'nANCFnodes':240,'functionData':functionData},
            #investigate different step sizes
            {'tEnd':2.45*f, 'nANCFnodes':240, 'stepSize':1e-5, 'functionData':functionData},
            {'tEnd':2.45*f, 'nANCFnodes':240, 'stepSize':2e-5, 'functionData':functionData},
            {'tEnd':2.45*f, 'nANCFnodes':240, 'stepSize':10e-5, 'functionData':functionData},
            #investigate different end times (evaluate stress plots)
            #NEEDS to adapt file names!!!
            {'tEnd':0.01*f, 'nANCFnodes':240,'functionData':functionData}, #beginning
            {'tEnd':0.1*f,  'nANCFnodes':240,'functionData':functionData},
            {'tEnd':0.5*f,  'nANCFnodes':240,'functionData':functionData},
            {'tEnd':1.*f,   'nANCFnodes':240,'functionData':functionData}, #here, the torque starts, but still at zero
            {'tEnd':1.25*f, 'nANCFnodes':240,'functionData':functionData}, #half of torque
            {'tEnd':1.5*f,  'nANCFnodes':240,'functionData':functionData}, #final torque
            {'tEnd':2*f,    'nANCFnodes':240,'functionData':functionData},
            #variation of other parameters:
            {'tEnd':2.45*f, 'nANCFnodes':240, 'contactStiffnessPerArea':1e7, 'functionData':functionData},
            {'tEnd':2.45*f, 'nANCFnodes':240, 'frictionStiffnessPerArea':1e7, 'functionData':functionData},
            {'tEnd':2.45*f, 'nANCFnodes':240, 'nSegments':2, 'functionData':functionData},
            {'tEnd':2.45*f, 'nANCFnodes':240, 'dryFriction':1, 'functionData':functionData},
            ]

        start_time = time.time()
        
        values = ProcessParameterList(parameterFunction=ParameterFunction, 
                             parameterList = parList,
                             debugMode=False,
                             addComputationIndex=runParallel,
                             useMultiProcessing=runParallel,
                             #numberOfThreads=4,
                             showProgress=True,)
        print("--- %s seconds ---" % (time.time() - start_time))
        print('values=', values) 
    elif performSingleSimulation:

        ParameterFunction({'tEnd':2.45, 'nANCFnodes':120, 'functionData':{'useGraphics':True}} )
        print("Perform single simulation")

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Section 8: Plot all figures
if performPlots:
    #fileDir = 'solution_new/'
    fileDir = 'solution/'
    #figureDir = 'figures/ESR8_'
    figureDir = 'figures_nosync/ESR8_'
    #parse file classifier and reconstruct values
    dTypes = {
              'tt':'end time',
              'hh':'step size',
              'nn': 'number of elements',
              'ns': 'number of segments',
              'nn': '$n_e$',
              'ns': '$n_{cs}$',
              'cs': '$k_c$',
              'fs': '$\mu_k$',
              'ib': 'improved belt model',
              'df': '$\mu$',
              }
    def FileClassifier2Data(fileClass):
        listParam = fileClass.split('-')
        print('listParam=',listParam[1:-1])
        d = dict()
        for x in listParam[1:-1]:
            tt = x[0:2]
            val = int(x[2:])
            if tt == 'tt': val = val*0.01
            if tt == 'hh': val = round(val*1e-6,6)
            if tt == 'nn': val *= 60
            if tt == 'cs': val = val*10 #MN #val*1e7
            if tt == 'fs': val = val*10 #MN #val*1e7
            if tt == 'df': val = val/10.
            d[tt] = val
        return d

    def FC2S(fc, fileClass):
        if fileClass == '-tt245-hh10-nn8-ns4-cs10-fs10-ib1-df5-':
            return 'reference solution'
        else:
            return dTypes[fc]+'='+str(FileClassifier2Data(fileClass)[fc])

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from exudyn.plot import PlotSensor, PlotSensorDefaults

    PlotSensorDefaults().sizeInches = [9,6]
    PlotSensorDefaults().fontSize = 15 #use 14; 12 is too small in .PDF


    PlotSensor(mbs, closeAll=False)
    
    caseList=[1,2,3,4]
    #caseList=[1,4] #0 are only tests
    for iCase in caseList:
        varLabels = []
        if iCase == 0:
            figureClass = 'varTest'
            variations = [
                          # '-tt90-hh100-nn1-ns4-cs10-fs10-ib1-df5-',
                          '-tt245-hh50-nn2-ns4-cs10-fs10-ib1-df5-',
                          ]
            for fileClass in variations:
                varLabels += [FC2S('tt',fileClass)]
        if iCase == 1:
            figureClass = 'varElem'
            variations = ['-tt245-hh50-nn1-ns4-cs10-fs10-ib1-df5-',
                          '-tt245-hh50-nn2-ns4-cs10-fs10-ib1-df5-', #nominal simulation
                          '-tt245-hh50-nn4-ns4-cs10-fs10-ib1-df5-', 
                          '-tt245-hh10-nn8-ns4-cs10-fs10-ib1-df5-', #reference solution
                          ]
            for fileClass in variations:
                varLabels += [FC2S('nn',fileClass)]
        elif iCase == 2:
            figureClass = 'varStepSize'
            variations = ['-tt245-hh10-nn4-ns4-cs10-fs10-ib1-df5-',
                          '-tt245-hh20-nn4-ns4-cs10-fs10-ib1-df5-',
                          '-tt245-hh50-nn4-ns4-cs10-fs10-ib1-df5-',
                          '-tt245-hh100-nn4-ns4-cs10-fs10-ib1-df5-',
                          ]
            for fileClass in variations:
                varLabels += [FC2S('hh',fileClass)]
        elif iCase == 3:
            figureClass = 'varPar'
            variations = ['-tt245-hh50-nn4-ns4-cs1-fs10-ib1-df5-',
                          '-tt245-hh50-nn4-ns4-cs10-fs1-ib1-df5-',
                          '-tt245-hh50-nn4-ns4-cs10-fs10-ib1-df10-',
                          '-tt245-hh50-nn4-ns4-cs10-fs10-ib1-df5-',
                          #'-tt245-hh50-nn4-ns2-cs10-fs10-ib1-df5-',
                          '-tt245-hh50-nn4-ns8-cs10-fs10-ib1-df5-',
                          ]
            for fileClass in variations:
                varLabels += [FC2S('ns',fileClass)+', '+
                              FC2S('df',fileClass)+', '+
                              FC2S('cs',fileClass)+'MN, '+
                              FC2S('fs',fileClass)+'MN'
                              ]
        elif iCase == 4:
            figureClass = 'varEndTime'
            variations = ['-tt1-hh50-nn4-ns4-cs10-fs10-ib1-df5-',
                          '-tt50-hh50-nn4-ns4-cs10-fs10-ib1-df5-',
                          '-tt100-hh50-nn4-ns4-cs10-fs10-ib1-df5-',
                          '-tt125-hh50-nn4-ns4-cs10-fs10-ib1-df5-',
                          '-tt150-hh50-nn4-ns4-cs10-fs10-ib1-df5-',
                          '-tt200-hh50-nn4-ns4-cs10-fs10-ib1-df5-',
                          '-tt245-hh50-nn4-ns4-cs10-fs10-ib1-df5-',
                          ]
            for fileClass in variations:
                varLabels += [FC2S('tt',fileClass)]
            
    
    
        beamVelFiles = []
        pulleyAngVel0Files = []
        pulleyAngVel1Files = []
        pulleyTorque0Files = []
        pulleyTorque1Files = []

        for fileClass in variations:
            beamVelFiles += [fileDir+'beamVel'+fileClass+'.txt']
            pulleyAngVel0Files += [fileDir+'wheel0angVel'+fileClass+'.txt']
            pulleyAngVel1Files += [fileDir+'wheel1angVel'+fileClass+'.txt']
            pulleyTorque0Files += [fileDir+'torquePulley0'+fileClass+'.txt']
            pulleyTorque1Files += [fileDir+'torquePulley1'+fileClass+'.txt']

    
            
        PlotSensor(mbs, sensorNumbers=beamVelFiles, components=0, labels=varLabels, 
                   xLabel='axial position (m)', yLabel='axial velocity (m/s)',
                   fileName=figureDir+figureClass+'AxialVelocities.pdf')
    
        beamForceFiles = []

        for fileClass in variations:
            beamForceFiles += [fileDir+'beamForce'+fileClass+'.txt']

    
        PlotSensor(mbs, sensorNumbers=beamForceFiles, components=0, labels=varLabels, colorCodeOffset=0,
                    xLabel='axial position (m)', yLabel='beam axial force (N)',
                    fileName=figureDir+figureClass+'AxialForces.pdf')
    
        contactForcesFiles = []
        contactForcesLabels = []
        for fileClass in variations:
            contactForcesFiles += [fileDir+'contactForces'+fileClass+'.txt']
  
    
    
        PlotSensor(mbs, sensorNumbers=contactForcesFiles, components=0, labels=varLabels, colorCodeOffset=0,
                    xLabel='axial position (m)', yLabel='contact tangential stress (N/m$^2$)',
                    fileName=figureDir+figureClass+'TangentialStress.pdf')
        
        PlotSensor(mbs, sensorNumbers=contactForcesFiles, components=1, labels=varLabels, 
                   colorCodeOffset=0+0*len(contactForcesFiles),newFigure=True,
                   xLabel='axial position (m)', yLabel='contact normal stress (N/m$^2$)',
                   fileName=figureDir+figureClass+'NormalStress.pdf')

        PlotSensor(mbs, sensorNumbers=pulleyAngVel1Files+[pulleyAngVel0Files[0]], components=2, labels=varLabels+['driven pulley $P_1$'], 
                   colorCodeOffset=0*len(pulleyAngVel1Files),newFigure=True,
                   xLabel='time (s)', yLabel='angular velocity (1/s)',
                   fileName=figureDir+figureClass+'Pulley2AngVel.pdf')

        PlotSensor(mbs, sensorNumbers=pulleyTorque0Files +[pulleyTorque1Files[0]], components=0, labels=varLabels+['pulley $P_2$'], 
                   colorCodeOffset=0*len(pulleyAngVel1Files),newFigure=True,
                   xLabel='time (s)', yLabel='torque (Nm)',
                   fileName=figureDir+figureClass+'Pulley1Torque.pdf')

        PlotSensor(mbs, closeAll=False)


    