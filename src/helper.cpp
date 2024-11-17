//
// Created by Wang Tai on 2024/10/20.
//

#include "helper.hpp"
#include <json/json.h>
#include "pipe_layout_generator.hpp"

void parseCurveInfo(const Json::Value& curveJson, CurveInfo& curve) {
    // Parse StartPoint
    const Json::Value& startPointJson = curveJson["StartPoint"];
    curve.StartPoint = Point{
        startPointJson["x"].asDouble(),
        startPointJson["y"].asDouble(),
        startPointJson["z"].asDouble()
    };

    // Parse EndPoint
    const Json::Value& endPointJson = curveJson["EndPoint"];
    curve.EndPoint = Point{
        endPointJson["x"].asDouble(),
        endPointJson["y"].asDouble(),
        endPointJson["z"].asDouble()
    };

    // Parse optional fields if they exist
    if (curveJson.isMember("MidPoint")) {
        const Json::Value& midPointJson = curveJson["MidPoint"];
        curve.MidPoint = Point{
            midPointJson["x"].asDouble(),
            midPointJson["y"].asDouble(),
            midPointJson["z"].asDouble()
        };
    }

    if (curveJson.isMember("Center")) {
        const Json::Value& centerJson = curveJson["Center"];
        curve.Center = Point{
            centerJson["x"].asDouble(),
            centerJson["y"].asDouble(),
            centerJson["z"].asDouble()
        };
    }

    curve.Radius = curveJson.get("Radius", 0.0).asDouble();
    curve.StartAngle = curveJson.get("StartAngle", 0.0).asDouble();
    curve.EndAngle = curveJson.get("EndAngle", 0.0).asDouble();
    curve.ColorIndex = curveJson.get("ColorIndex", 0).asInt();
    curve.CurveType = curveJson.get("CurveType", 0).asInt();
}
std::string planToJson(const HeatingDesign& plan) {
    Json::Value root;
    // Convert HeatingDesign to JSON
    for (const auto& heatingCoil : plan.HeatingCoils) {
        Json::Value heatingCoilJson;
        heatingCoilJson["LevelName"] = heatingCoil.LevelName;
        heatingCoilJson["LevelNo"] = heatingCoil.LevelNo;
        heatingCoilJson["LevelDesc"] = heatingCoil.LevelDesc;
        heatingCoilJson["HouseName"] = heatingCoil.HouseName;

        // Convert Expansions
        Json::Value expansionsJson(Json::arrayValue);
        for (const auto& expansion : heatingCoil.Expansions) {
            Json::Value expansionJson;
            expansionJson["StartPoint"]["x"] = expansion.StartPoint.x;
            expansionJson["StartPoint"]["y"] = expansion.StartPoint.y;
            expansionJson["StartPoint"]["z"] = expansion.StartPoint.z;
            expansionJson["EndPoint"]["x"] = expansion.EndPoint.x;
            expansionJson["EndPoint"]["y"] = expansion.EndPoint.y;
            expansionJson["EndPoint"]["z"] = expansion.EndPoint.z;
            expansionJson["ColorIndex"] = expansion.ColorIndex;
            expansionJson["CurveType"] = expansion.CurveType;
            expansionsJson.append(expansionJson);
        }
        heatingCoilJson["Expansions"] = expansionsJson;

        // Convert CollectorCoils
        Json::Value collectorCoilsJson(Json::arrayValue);
        for (const auto& collectorCoil : heatingCoil.CollectorCoils) {
            Json::Value collectorCoilJson;
            collectorCoilJson["CollectorName"] = collectorCoil.CollectorName;
            collectorCoilJson["Loops"] = collectorCoil.Loops;

            // Convert CoilLoops
            Json::Value coilLoopsJson(Json::arrayValue);
            for (const auto& coilLoop : collectorCoil.CoilLoops) {
                Json::Value coilLoopJson;
                coilLoopJson["Length"] = coilLoop.Length;
                coilLoopJson["Curvity"] = coilLoop.Curvity;

                // Convert Areas
                Json::Value areasJson(Json::arrayValue);
                for (const auto& area : coilLoop.Areas) {
                    Json::Value areaJson;
                    areaJson["AreaName"] = area.AreaName;
                    areasJson.append(areaJson);
                }
                coilLoopJson["Areas"] = areasJson;

                // Convert Path
                Json::Value pathJson(Json::arrayValue);
                for (const auto& jLine : coilLoop.Path) {
                    Json::Value jLineJson;
                    jLineJson["StartPoint"]["x"] = jLine.StartPoint.x;
                    jLineJson["StartPoint"]["y"] = jLine.StartPoint.y;
                    jLineJson["StartPoint"]["z"] = jLine.StartPoint.z;
                    jLineJson["EndPoint"]["x"] = jLine.EndPoint.x;
                    jLineJson["EndPoint"]["y"] = jLine.EndPoint.y;
                    jLineJson["EndPoint"]["z"] = jLine.EndPoint.z;
                    jLineJson["ColorIndex"] = jLine.ColorIndex;
                    jLineJson["CurveType"] = jLine.CurveType;
                    pathJson.append(jLineJson);
                }
                coilLoopJson["Path"] = pathJson;

                coilLoopsJson.append(coilLoopJson);
            }
            collectorCoilJson["CoilLoops"] = coilLoopsJson;

            // Convert Deliverys
            Json::Value deliverysJson(Json::arrayValue);
            for (const auto& delivery : collectorCoil.Deliverys) {
                Json::Value deliveryJson(Json::arrayValue);
                for (const auto& jLine : delivery) {
                    Json::Value jLineJson;
                    jLineJson["StartPoint"]["x"] = jLine.StartPoint.x;
                    jLineJson["StartPoint"]["y"] = jLine.StartPoint.y;
                    jLineJson["StartPoint"]["z"] = jLine.StartPoint.z;
                    jLineJson["EndPoint"]["x"] = jLine.EndPoint.x;
                    jLineJson["EndPoint"]["y"] = jLine.EndPoint.y;
                    jLineJson["EndPoint"]["z"] = jLine.EndPoint.z;
                    jLineJson["ColorIndex"] = jLine.ColorIndex;
                    jLineJson["CurveType"] = jLine.CurveType;
                    deliveryJson.append(jLineJson);
                }
                deliverysJson.append(deliveryJson);
            }
            collectorCoilJson["Deliverys"] = deliverysJson;

            collectorCoilsJson.append(collectorCoilJson);
        }
        heatingCoilJson["CollectorCoils"] = collectorCoilsJson;

        root["HeatingCoils"].append(heatingCoilJson);
    }
    Json::FastWriter writer;
    return writer.write(root);
}


CombinedData parseJsonData(const std::string& arDesignJson, const std::string& inputDataJson){
    CombinedData combinedData;

    // Parse ARDesign.json
    Json::Value arDesignRoot;
    Json::Reader arDesignReader;
    if (!arDesignReader.parse(arDesignJson, arDesignRoot)) {
        throw std::runtime_error("Failed to parse ARDesign.json");
    }

    // Parse Floors
    const Json::Value& floorsJson = arDesignRoot["Floors"];
    for (const auto& floorJson : floorsJson) {
        Floor floor;
        floor.Name = floorJson["Name"].asString();
        floor.Num = floorJson["Num"].asString();
        floor.LevelHeight = floorJson["LevelHeight"].asDouble();

        // Parse Construction
        const Json::Value& constructionJson = floorJson["Construction"];
        
        // Parse HouseTypes
        const Json::Value& houseTypesJson = constructionJson["HouseType"];
        for (const auto& houseTypeJson : houseTypesJson) {
            HouseType houseType;
            houseType.houseName = houseTypeJson["houseName"].asString();
            for (const auto& roomName : houseTypeJson["RoomNames"])
                houseType.RoomNames.push_back(roomName.asString());
            for (const auto& pointJson : houseTypeJson["Boundary"])
                houseType.Boundary.push_back(Point{pointJson["x"].asDouble(), pointJson["y"].asDouble(), pointJson["z"].asDouble()});
            floor.construction.houseTypes.push_back(houseType);
        }

        // Parse Rooms
        const Json::Value& roomsJson = constructionJson["Room"];
        for (const auto& roomJson : roomsJson) {
            Room room;
            room.Guid = roomJson["Guid"].asString();
            room.Name = roomJson["Name"].asString();
            room.NameType = roomJson["NameType"].asString();
            for (const auto& doorId : roomJson["DoorIds"])
                room.DoorIds.push_back(doorId.asString());
            for (const auto& jcwGuidName : roomJson["JCWGuidNames"])
                room.JCWGuidNames.push_back(jcwGuidName.asString());
            for (const auto& wallName : roomJson["WallNames"])
                room.WallNames.push_back(wallName.asString());
            for (const auto& boundaryJson : roomJson["Boundary"]) {
                CurveInfo curve;
                curve.StartPoint = Point{
                    boundaryJson["StartPoint"]["x"].asDouble(),
                    boundaryJson["StartPoint"]["y"].asDouble(),
                    boundaryJson["StartPoint"]["z"].asDouble()
                };
                curve.EndPoint = Point{
                    boundaryJson["EndPoint"]["x"].asDouble(),
                    boundaryJson["EndPoint"]["y"].asDouble(),
                    boundaryJson["EndPoint"]["z"].asDouble()
                };
                curve.ColorIndex = boundaryJson["ColorIndex"].asInt();
                curve.CurveType = boundaryJson["CurveType"].asInt();
                
                if (boundaryJson.isMember("MidPoint")) {
                    curve.MidPoint = Point{
                        boundaryJson["MidPoint"]["x"].asDouble(),
                        boundaryJson["MidPoint"]["y"].asDouble(),
                        boundaryJson["MidPoint"]["z"].asDouble()
                    };
                }
                if (boundaryJson.isMember("Center")) {
                    curve.Center = Point{
                        boundaryJson["Center"]["x"].asDouble(),
                        boundaryJson["Center"]["y"].asDouble(),
                        boundaryJson["Center"]["z"].asDouble()
                    };
                }
                if (boundaryJson.isMember("Radius")) {
                    curve.Radius = boundaryJson["Radius"].asDouble();
                }
                if (boundaryJson.isMember("StartAngle")) {
                    curve.StartAngle = boundaryJson["StartAngle"].asDouble();
                }
                if (boundaryJson.isMember("EndAngle")) {
                    curve.EndAngle = boundaryJson["EndAngle"].asDouble();
                }

                room.Boundary.push_back(curve);
            }
            floor.construction.rooms.push_back(room);
        }

        // Parse JCWs
        const Json::Value& jcwsJson = constructionJson["JCW"];
        for (const auto& jcwJson : jcwsJson) {
            JCW jcw;
            jcw.GuidName = jcwJson["GuidName"].asString();
            jcw.Type = jcwJson["Type"].asInt();
            jcw.Name = jcwJson["Name"].asString();
            
            // Parse CenterPoint
            const Json::Value& centerPointJson = jcwJson["CenterPoint"];
            jcw.CenterPoint = Point{
                centerPointJson["x"].asDouble(),
                centerPointJson["y"].asDouble(),
                centerPointJson["z"].asDouble()
            };

            // Parse ShowCurves, MaxBoundaryCurves, and BoundaryLines
            if (jcwJson.isMember("ShowCurves")) {
                for (const auto& curveJson : jcwJson["ShowCurves"]) {
                    CurveInfo curve;
                    parseCurveInfo(curveJson, curve);
                    jcw.ShowCurves.push_back(curve);
                }
            }

            if (jcwJson.isMember("MaxBoundaryCurves")) {
                for (const auto& curveJson : jcwJson["MaxBoundaryCurves"]) {
                    CurveInfo curve;
                    parseCurveInfo(curveJson, curve);
                    jcw.MaxBoundaryCurves.push_back(curve);
                }
            }

            if (jcwJson.isMember("BoundaryLines")) {
                for (const auto& curveJson : jcwJson["BoundaryLines"]) {
                    CurveInfo curve;
                    parseCurveInfo(curveJson, curve);
                    jcw.BoundaryLines.push_back(curve);
                }
            }

            // Parse Parameters
            const Json::Value& paramsJson = jcwJson["Parameters"];
            for (const auto& paramName : paramsJson.getMemberNames()) {
                jcw.Parameters[paramName] = paramsJson[paramName].asString();
            }

            jcw.IsBlockLayer = jcwJson["IsBlockLayer"].asBool();
            floor.construction.jcws.push_back(jcw);
        }

        // Parse Doors
        const Json::Value& doorsJson = constructionJson["Door"];
        for (const auto& doorJson : doorsJson) {
            DoorAndWindow door;
            door.Guid = doorJson["Guid"].asString();
            door.FamilyName = doorJson["FamilyName"].asString();
            door.Name = doorJson["Name"].asString();
            door.DoorType = doorJson["DoorType"].asInt();
            door.HostWall = doorJson["HostWall"].asString();
            
            // Parse Location and Size
            const Json::Value& locationJson = doorJson["Location"];
            door.Location = Point{
                locationJson["x"].asDouble(),
                locationJson["y"].asDouble(),
                locationJson["z"].asDouble()
            };

            const Json::Value& sizeJson = doorJson["DoorSize"];
            door.Size = Size{
                sizeJson["Height"].asDouble(),
                sizeJson["Width"].asDouble(),
                sizeJson["Thickness"].asDouble()
            };

            // Parse FlipFaceNormal and FlipHandNormal
            const Json::Value& flipFaceJson = doorJson["FlipFaceNormal"];
            door.FlipFaceNormal = Point{
                flipFaceJson["x"].asDouble(),
                flipFaceJson["y"].asDouble(),
                flipFaceJson["z"].asDouble()
            };

            // Additional door properties
            door.Number = doorJson["Number"].asString();
            door.IsHaveShutter = doorJson["IsHaveShutter"].asBool();
            door.MortarThickness = doorJson["MortarThickness"].asDouble();
            door.AirArea = doorJson["AirArea"].asDouble();
            door.OpenDirection = doorJson["OpenDirection"].asString();
            door.FireClass = doorJson["FireClass"].asString();
            
            floor.construction.doorAndWindows.push_back(door);
        }

        // Parse Windows
        const Json::Value& windowsJson = constructionJson["Window"];
        for (const auto& windowJson : windowsJson) {
            DoorAndWindow window;
            window.Guid = windowJson["Guid"].asString();
            window.FamilyName = windowJson["FamilyName"].asString();
            window.Name = windowJson["Name"].asString();
            window.Type = windowJson["Type"].asString();
            window.HostWall = windowJson["HostWall"].asString();
            
            // Parse Location
            const Json::Value& locationJson = windowJson["Location"];
            window.Location = Point{
                locationJson["x"].asDouble(),
                locationJson["y"].asDouble(),
                locationJson["z"].asDouble()
            };

            window.BottomHeight = windowJson["BottomHeight"].asDouble();
            window.TopHeight = windowJson["TopHeight"].asDouble();
            
            // Parse Window Style
            const Json::Value& styleJson = windowJson["Style"];
            window.Style.StyleId = styleJson["StyleId"].asInt();
            window.Style.StyleName = styleJson["StyleName"].asString();
            window.Style.SSMHeight = styleJson["SSMHeight"].asDouble();
            window.Style.SSMWidth = styleJson["SSMWidth"].asDouble();
            
            window.IsVisible = windowJson["IsVisible"].asBool();
            window.IsMirror = windowJson["IsMirror"].asBool();
            window.IsFire = windowJson["IsFire"].asBool();
            
            floor.construction.doorAndWindows.push_back(window);
        }

        combinedData.arDesign.Floor.push_back(floor);
    }

    // Parse inputData.json
    Json::Value inputDataRoot;
    Json::Reader inputDataReader;
    if (!inputDataReader.parse(inputDataJson, inputDataRoot)) {
        throw std::runtime_error("Failed to parse inputData.json");
    }

    // Parse AssistData
    const Json::Value& assistDataJson = inputDataRoot["AssistData"];
    for (const auto& collectorJson : assistDataJson["AssistCollectors"]) {
        AssistCollector collector;
        collector.Id = collectorJson["Id"].asString();
        collector.Loc = Point{collectorJson["Loc"]["x"].asDouble(), collectorJson["Loc"]["y"].asDouble(), collectorJson["Loc"]["z"].asDouble()};
        collector.LevelName = collectorJson["LevelName"].asString();
        for (const auto& boundaryJson : collectorJson["Boundaries"]) {
            AssistCollector::Boundary boundary;
            boundary.Offset = boundaryJson["Offset"].asDouble();
            for (const auto& borderJson : boundaryJson["Borders"]) {
                Border border;
                border.StartPoint = Point{borderJson["StartPoint"]["x"].asDouble(), borderJson["StartPoint"]["y"].asDouble(), borderJson["StartPoint"]["z"].asDouble()};
                border.EndPoint = Point{borderJson["EndPoint"]["x"].asDouble(), borderJson["EndPoint"]["y"].asDouble(), borderJson["EndPoint"]["z"].asDouble()};
                border.ColorIndex = borderJson["ColorIndex"].asInt();
                border.CurveType = borderJson["CurveType"].asInt();
                boundary.Borders.push_back(border);
            }
            collector.Boundaries.push_back(boundary);
        }
        combinedData.inputData.assistData.AssistCollectors.push_back(collector);
    }

    // Parse WebData
    const Json::Value& webDataJson = inputDataRoot["WebData"];
    combinedData.inputData.webData.ImbalanceRatio = webDataJson["ImbalanceRatio"].asInt();
    combinedData.inputData.webData.JointPipeSpan = webDataJson["JointPipeSpan"].asDouble();
    combinedData.inputData.webData.DenseAreaWallSpan = webDataJson["DenseAreaWallSpan"].asDouble();
    combinedData.inputData.webData.DenseAreaSpanLess = webDataJson["DenseAreaSpanLess"].asDouble();

    // Parse LoopSpanSet, ObsSpanSet, DeliverySpanSet, PipeSpanSet, ElasticSpanSet, and FuncRooms
    // Parse LoopSpanSet
    const Json::Value& loopSpanSetJson = webDataJson["LoopSpanSet"];
    for (const auto& loopSpanJson : loopSpanSetJson) {
        LoopSpan loopSpan;
        loopSpan.TypeName = loopSpanJson["TypeName"].asString();
        loopSpan.MinSpan = loopSpanJson["MinSpan"].asDouble();
        loopSpan.MaxSpan = loopSpanJson["MaxSpan"].asDouble();
        loopSpan.Curvity = loopSpanJson["Curvity"].asDouble();
        combinedData.inputData.webData.LoopSpanSet.push_back(loopSpan);
    }

    // Parse ObsSpanSet
    const Json::Value& obsSpanSetJson = webDataJson["ObsSpanSet"];
    for (const auto& obsSpanJson : obsSpanSetJson) {
        ObstacleSpan obsSpan;
        obsSpan.ObsName = obsSpanJson["ObsName"].asString();
        obsSpan.MinSpan = obsSpanJson["MinSpan"].asDouble();
        obsSpan.MaxSpan = obsSpanJson["MaxSpan"].asDouble();
        combinedData.inputData.webData.ObsSpanSet.push_back(obsSpan);
    }

    // Parse DeliverySpanSet
    const Json::Value& deliverySpanSetJson = webDataJson["DeliverySpanSet"];
    for (const auto& deliverySpanJson : deliverySpanSetJson) {
        ObstacleSpan deliverySpan;
        deliverySpan.ObsName = deliverySpanJson["ObsName"].asString();
        deliverySpan.MinSpan = deliverySpanJson["MinSpan"].asDouble();
        deliverySpan.MaxSpan = deliverySpanJson["MaxSpan"].asDouble();
        combinedData.inputData.webData.DeliverySpanSet.push_back(deliverySpan);
    }

    // Parse PipeSpanSet
    const Json::Value& pipeSpanSetJson = webDataJson["PipeSpanSet"];
    for (const auto& pipeSpanJson : pipeSpanSetJson) {
        PipeSpanSet pipeSpan;
        pipeSpan.LevelDesc = pipeSpanJson["LevelDesc"].asString();
        pipeSpan.FuncName = pipeSpanJson["FuncName"].asString();
        for (const auto& direction : pipeSpanJson["Directions"])
            pipeSpan.Directions.push_back(direction.asString());
        pipeSpan.ExterWalls = pipeSpanJson["ExterWalls"].asInt();
        pipeSpan.PipeSpan = pipeSpanJson["PipeSpan"].asDouble();
        combinedData.inputData.webData.PipeSpanSet.push_back(pipeSpan);
    }

    // Parse ElasticSpanSet
    const Json::Value& elasticSpanSetJson = webDataJson["ElasticSpanSet"];
    for (const auto& elasticSpanJson : elasticSpanSetJson) {
        ElasticSpan elasticSpan;
        elasticSpan.FuncName = elasticSpanJson["FuncName"].asString();
        elasticSpan.PriorSpan = elasticSpanJson["PriorSpan"].asDouble();
        elasticSpan.MinSpan = elasticSpanJson["MinSpan"].asDouble();
        elasticSpan.MaxSpan = elasticSpanJson["MaxSpan"].asDouble();
        combinedData.inputData.webData.ElasticSpanSet.push_back(elasticSpan);
    }

    // Parse FuncRooms
    const Json::Value& funcRoomsJson = webDataJson["FuncRooms"];
    for (const auto& funcRoomJson : funcRoomsJson) {
        FuncRoom funcRoom;
        funcRoom.FuncName = funcRoomJson["FuncName"].asString();
        for (const auto& roomName : funcRoomJson["RoomNames"])
            funcRoom.RoomNames.push_back(roomName.asString());
        combinedData.inputData.webData.FuncRooms.push_back(funcRoom);
    }

    return combinedData;
}

HeatingDesign generatePipePlan(const CombinedData& combinedData){
    HeatingDesign heatingDesign;

    // Iterate through each floor
    for (const auto& floor : combinedData.arDesign.Floor) {
        HeatingCoil heatingCoil;
        heatingCoil.LevelName = floor.Name;
        heatingCoil.LevelNo = std::stoi(floor.Num);
        heatingCoil.LevelDesc = "Floor " + floor.Num;

        // Iterate through each house type
        for (const auto& houseType : floor.construction.houseTypes) {
            heatingCoil.HouseName = houseType.houseName;

            // Call the pipe layout generation function
            CollectorCoil collectorCoil = generatePipeLayout(houseType, combinedData.inputData.webData);

            // Add the generated CollectorCoil to HeatingCoil
            heatingCoil.CollectorCoils.push_back(collectorCoil);
        }

        // Add the generated HeatingCoil to HeatingDesign
        heatingDesign.HeatingCoils.push_back(heatingCoil);
    }

    return heatingDesign;
}

