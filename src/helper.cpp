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

ARDesign parseARDesign(const std::string& arDesignJson) {
    ARDesign design;
    Json::Value j;
    Json::Reader reader;
    
    if (!reader.parse(arDesignJson, j)) {
        throw std::runtime_error("Failed to parse AR Design JSON");
    }
    
    // Parse Floor array
    if (j.isMember("Floor") && j["Floor"].isArray()) {
        for (const auto& floorJson : j["Floor"]) {
            Floor floor;
            
            // Parse floor basic info
            floor.Name = floorJson.get("Name", "").asString();
            floor.Num = floorJson.get("Num", "").asString();
            floor.LevelHeight = floorJson.get("LevelHeight", 0.0).asDouble();
            
            // Parse Construction
            if (floorJson.isMember("construction")) {
                const auto& constJson = floorJson["construction"];
                
                // Parse HouseTypes
                if (constJson.isMember("houseTypes")) {
                    for (const auto& htJson : constJson["houseTypes"]) {
                        HouseType ht;
                        ht.houseName = htJson.get("houseName", "").asString();
                        
                        // Parse RoomNames array
                        if (htJson.isMember("RoomNames") && htJson["RoomNames"].isArray()) {
                            for (const auto& roomName : htJson["RoomNames"]) {
                                ht.RoomNames.push_back(roomName.asString());
                            }
                        }
                        
                        // Parse Boundary points
                        if (htJson.isMember("Boundary")) {
                            for (const auto& ptJson : htJson["Boundary"]) {
                                Point pt{};
                                pt.x = ptJson.get("X", 0.0).asDouble();
                                pt.y = ptJson.get("Y", 0.0).asDouble();
                                pt.z = ptJson.get("Z", 0.0).asDouble();
                                ht.Boundary.push_back(pt);
                            }
                        }
                        floor.construction.houseTypes.push_back(ht);
                    }
                }
                
                // Parse Rooms
                if (constJson.isMember("rooms")) {
                    for (const auto& roomJson : constJson["rooms"]) {
                        Room room;
                        room.Guid = roomJson.get("Guid", "").asString();
                        room.Name = roomJson.get("Name", "").asString();
                        room.NameType = roomJson.get("NameType", "").asString();
                        
                        // Parse arrays
                        if (roomJson.isMember("DoorIds") && roomJson["DoorIds"].isArray()) {
                            for (const auto& id : roomJson["DoorIds"]) {
                                room.DoorIds.push_back(id.asString());
                            }
                        }
                        if (roomJson.isMember("JCWGuidNames") && roomJson["JCWGuidNames"].isArray()) {
                            for (const auto& name : roomJson["JCWGuidNames"]) {
                                room.JCWGuidNames.push_back(name.asString());
                            }
                        }
                        if (roomJson.isMember("WallNames") && roomJson["WallNames"].isArray()) {
                            for (const auto& name : roomJson["WallNames"]) {
                                room.WallNames.push_back(name.asString());
                            }
                        }
                        
                        room.BlCreateRoom = roomJson.get("BlCreateRoom", 0).asInt();
                        
                        // Parse Boundary curves
                        if (roomJson.isMember("Boundary")) {
                            for (const auto& curveJson : roomJson["Boundary"]) {
                                CurveInfo curve;
                                parseCurveInfo(curveJson, curve);
                                room.Boundary.push_back(curve);
                            }
                        }
                        floor.construction.rooms.push_back(room);
                    }
                }
                
                // Parse JCWs
                if (constJson.isMember("jcws")) {
                    for (const auto& jcwJson : constJson["jcws"]) {
                        JCW jcw;
                        jcw.GuidName = jcwJson.get("GuidName", "").asString();
                        jcw.Type = jcwJson.get("Type", 0).asInt();
                        jcw.Name = jcwJson.get("Name", "").asString();
                        
                        // Parse CenterPoint
                        if (jcwJson.isMember("CenterPoint")) {
                            const auto& centerJson = jcwJson["CenterPoint"];
                            jcw.CenterPoint.x = centerJson.get("X", 0.0).asDouble();
                            jcw.CenterPoint.y = centerJson.get("Y", 0.0).asDouble();
                            jcw.CenterPoint.z = centerJson.get("Z", 0.0).asDouble();
                        }
                        
                        // Parse BoundaryLines
                        if (jcwJson.isMember("BoundaryLines")) {
                            for (const auto& curveJson : jcwJson["BoundaryLines"]) {
                                CurveInfo curve;
                                parseCurveInfo(curveJson, curve);
                                jcw.BoundaryLines.push_back(curve);
                            }
                        }
                        floor.construction.jcws.push_back(jcw);
                    }
                }
                
                // Parse Doors
                if (constJson.isMember("door")) {
                    for (const auto& doorJson : constJson["door"]) {
                        Door door;
                        door.Guid = doorJson.get("Guid", "").asString();
                        door.FamilyName = doorJson.get("FamilyName", "").asString();
                        door.DoorType = doorJson.get("DoorType", 0).asInt();
                        door.HostWall = doorJson.get("HostWall", "").asString();
                        
                        // Parse Location
                        if (doorJson.isMember("Location")) {
                            const auto& locJson = doorJson["Location"];
                            door.Location.x = locJson.get("X", 0.0).asDouble();
                            door.Location.y = locJson.get("Y", 0.0).asDouble();
                            door.Location.z = locJson.get("Z", 0.0).asDouble();
                        }
                        
                        // Parse Size
                        if (doorJson.isMember("Size")) {
                            const auto& sizeJson = doorJson["Size"];
                            door.Size.Height = sizeJson.get("Height", 0.0).asDouble();
                            door.Size.Width = sizeJson.get("Width", 0.0).asDouble();
                            door.Size.Thickness = sizeJson.get("Thickness", 0.0).asDouble();
                        }
                        
                        // Parse Direction Normals
                        if (doorJson.isMember("FlipFaceNormal")) {
                            const auto& normalJson = doorJson["FlipFaceNormal"];
                            door.FlipFaceNormal.x = normalJson.get("X", 0.0).asDouble();
                            door.FlipFaceNormal.y = normalJson.get("Y", 0.0).asDouble();
                            door.FlipFaceNormal.z = normalJson.get("Z", 0.0).asDouble();
                        }
                        if (doorJson.isMember("FlipHandNormal")) {
                            const auto& normalJson = doorJson["FlipHandNormal"];
                            door.FlipHandNormal.x = normalJson.get("X", 0.0).asDouble();
                            door.FlipHandNormal.y = normalJson.get("Y", 0.0).asDouble();
                            door.FlipHandNormal.z = normalJson.get("Z", 0.0).asDouble();
                        }
                        
                        floor.construction.door.push_back(door);
                    }
                }
            }
            design.Floor.push_back(floor);
        }
    }
    return design;
}

// Helper function to parse inputData.json
InputData parseInputData(const std::string& inputDataJson) {
    InputData inputData;
    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(inputDataJson, root)) {
        throw std::runtime_error("Failed to parse inputData.json");
    }

    // Parse AssistData
    const Json::Value& assistDataJson = root["AssistData"];
    for (const auto& collectorJson : assistDataJson["AssistCollectors"]) {
        AssistCollector collector;
        collector.Id = collectorJson["Id"].asString();
        collector.Loc = Point{collectorJson["Loc"]["x"].asDouble(), collectorJson["Loc"]["y"].asDouble(), collectorJson["Loc"]["z"].asDouble()};
        collector.LevelName = collectorJson["LevelName"].asString();
        for (const auto& boundaryJson : collectorJson["Boundaries"]) {
            AssistCollector::Boundary boundary;
            boundary.Offset = boundaryJson["Offset"].asDouble();
            for (const auto& borderJson : boundaryJson["Borders"]) {
                Border border{};
                border.StartPoint = Point{borderJson["StartPoint"]["x"].asDouble(), borderJson["StartPoint"]["y"].asDouble(), borderJson["StartPoint"]["z"].asDouble()};
                border.EndPoint = Point{borderJson["EndPoint"]["x"].asDouble(), borderJson["EndPoint"]["y"].asDouble(), borderJson["EndPoint"]["z"].asDouble()};
                border.ColorIndex = borderJson["ColorIndex"].asInt();
                border.CurveType = borderJson["CurveType"].asInt();
                boundary.Borders.push_back(border);
            }
            collector.Boundaries.push_back(boundary);
        }
        inputData.assistData.AssistCollectors.push_back(collector);
    }

    // Parse WebData
    const Json::Value& webDataJson = root["WebData"];
    inputData.webData.ImbalanceRatio = webDataJson["ImbalanceRatio"].asInt();
    inputData.webData.JointPipeSpan = webDataJson["JointPipeSpan"].asDouble();
    inputData.webData.DenseAreaWallSpan = webDataJson["DenseAreaWallSpan"].asDouble();
    inputData.webData.DenseAreaSpanLess = webDataJson["DenseAreaSpanLess"].asDouble();

    // Parse LoopSpanSet, ObsSpanSet, DeliverySpanSet, PipeSpanSet, ElasticSpanSet, and FuncRooms
    // Parse LoopSpanSet
    const Json::Value& loopSpanSetJson = webDataJson["LoopSpanSet"];
    for (const auto& loopSpanJson : loopSpanSetJson) {
        LoopSpan loopSpan;
        loopSpan.TypeName = loopSpanJson["TypeName"].asString();
        loopSpan.MinSpan = loopSpanJson["MinSpan"].asDouble();
        loopSpan.MaxSpan = loopSpanJson["MaxSpan"].asDouble();
        loopSpan.Curvity = loopSpanJson["Curvity"].asDouble();
        inputData.webData.LoopSpanSet.push_back(loopSpan);
    }

    // Parse ObsSpanSet
    const Json::Value& obsSpanSetJson = webDataJson["ObsSpanSet"];
    for (const auto& obsSpanJson : obsSpanSetJson) {
        ObstacleSpan obsSpan;
        obsSpan.ObsName = obsSpanJson["ObsName"].asString();
        obsSpan.MinSpan = obsSpanJson["MinSpan"].asDouble();
        obsSpan.MaxSpan = obsSpanJson["MaxSpan"].asDouble();
        inputData.webData.ObsSpanSet.push_back(obsSpan);
    }

    // Parse DeliverySpanSet
    const Json::Value& deliverySpanSetJson = webDataJson["DeliverySpanSet"];
    for (const auto& deliverySpanJson : deliverySpanSetJson) {
        ObstacleSpan deliverySpan;
        deliverySpan.ObsName = deliverySpanJson["ObsName"].asString();
        deliverySpan.MinSpan = deliverySpanJson["MinSpan"].asDouble();
        deliverySpan.MaxSpan = deliverySpanJson["MaxSpan"].asDouble();
        inputData.webData.DeliverySpanSet.push_back(deliverySpan);
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
        inputData.webData.PipeSpanSet.push_back(pipeSpan);
    }

    // Parse ElasticSpanSet
    const Json::Value& elasticSpanSetJson = webDataJson["ElasticSpanSet"];
    for (const auto& elasticSpanJson : elasticSpanSetJson) {
        ElasticSpan elasticSpan;
        elasticSpan.FuncName = elasticSpanJson["FuncName"].asString();
        elasticSpan.PriorSpan = elasticSpanJson["PriorSpan"].asDouble();
        elasticSpan.MinSpan = elasticSpanJson["MinSpan"].asDouble();
        elasticSpan.MaxSpan = elasticSpanJson["MaxSpan"].asDouble();
        inputData.webData.ElasticSpanSet.push_back(elasticSpan);
    }

    // Parse FuncRooms
    const Json::Value& funcRoomsJson = webDataJson["FuncRooms"];
    for (const auto& funcRoomJson : funcRoomsJson) {
        FuncRoom funcRoom;
        funcRoom.FuncName = funcRoomJson["FuncName"].asString();
        for (const auto& roomName : funcRoomJson["RoomNames"])
            funcRoom.RoomNames.push_back(roomName.asString());
        inputData.webData.FuncRooms.push_back(funcRoom);
    }

    return inputData;
}

// Main parsing function that combines both
CombinedData parseJsonData(const std::string& arDesignJson, const std::string& inputDataJson) {
    CombinedData combinedData;
    
    try {
        combinedData.arDesign = parseARDesign(arDesignJson);
        combinedData.inputData = parseInputData(inputDataJson);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error parsing JSON: ") + e.what());
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

