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

auto planToJson(const HeatingDesign& plan) -> std::string {
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

// Add this before parseARDesign function
void parseRoom(const Json::Value& roomJson, Room& room) {
    room.Name = roomJson.get("Name", "").asString();
    room.Guid = roomJson.get("Guid", "").asString();
    room.NameType = roomJson.get("NameType", "").asString();
    room.BlCreateRoom = roomJson.get("BlCreateRoom", 0).asInt();

    // Parse DoorIds
    if (roomJson.isMember("DoorIds") && roomJson["DoorIds"].isArray()) {
        for (const auto& doorId : roomJson["DoorIds"]) {
            room.DoorIds.push_back(doorId.asString());
        }
    }

    // Parse JCWGuidNames
    if (roomJson.isMember("JCWGuidNames") && roomJson["JCWGuidNames"].isArray()) {
        for (const auto& jcwGuidName : roomJson["JCWGuidNames"]) {
            room.JCWGuidNames.push_back(jcwGuidName.asString());
        }
    }

    // Parse WallNames
    if (roomJson.isMember("WallNames") && roomJson["WallNames"].isArray()) {
        for (const auto& wallName : roomJson["WallNames"]) {
            room.WallNames.push_back(wallName.asString());
        }
    }

    // Parse Borders if they exist
    if (roomJson.isMember("Boundary") && roomJson["Boundary"].isArray()) {
        for (const auto& borderJson : roomJson["Boundary"]) {
            CurveInfo border;
            border.StartPoint = Point{
                borderJson["StartPoint"]["x"].asDouble(),
                borderJson["StartPoint"]["y"].asDouble(),
                borderJson["StartPoint"]["z"].asDouble()
            };
            border.EndPoint = Point{
                borderJson["EndPoint"]["x"].asDouble(),
                borderJson["EndPoint"]["y"].asDouble(),
                borderJson["EndPoint"]["z"].asDouble()
            };
            border.ColorIndex = borderJson.get("ColorIndex", 0).asInt();
            border.CurveType = borderJson.get("CurveType", 0).asInt();
            room.Boundary.push_back(border);
        }
    }
}

void parseJCW(const Json::Value& jcwJson, JCW& jcw) {
    // Parse basic properties
    jcw.GuidName = jcwJson.get("GuidName", "").asString();
    jcw.Type = jcwJson.get("Type", 0).asInt();
    jcw.Name = jcwJson.get("Name", "").asString();

    // Parse CenterPoint
    if (jcwJson.isMember("CenterPoint")) {
        const auto& centerPoint = jcwJson["CenterPoint"];
        jcw.CenterPoint = Point{
            centerPoint["x"].asDouble(),
            centerPoint["y"].asDouble(),
            centerPoint["z"].asDouble()
        };
    }

    // Parse BoundaryLines array
    if (jcwJson.isMember("BoundaryLines") && jcwJson["BoundaryLines"].isArray()) {
        for (const auto& lineJson : jcwJson["BoundaryLines"]) {
            CurveInfo curve;
            parseCurveInfo(lineJson, curve);
            jcw.BoundaryLines.push_back(curve);
        }
    }
}


void parseDoor(const Json::Value& doorJson, Door& door) {
    // Parse basic properties
    door.Guid = doorJson.get("Guid", "").asString();
    door.FamilyName = doorJson.get("FamilyName", "").asString();
    door.DoorType = doorJson.get("DoorType", "").asString();
    door.HostWall = doorJson.get("HostWall", "").asString();

    // Parse Location
    if (doorJson.isMember("Location")) {
        const auto& location = doorJson["Location"];
        door.Location = Point{
            location["x"].asDouble(),
            location["y"].asDouble(),
            location["z"].asDouble()
        };
    }

    // Parse Size
    if (doorJson.isMember("Size")) {
        const auto& size = doorJson["Size"];
        door.Size = Size{
            size.get("Height", 0.0).asDouble(),
            size.get("Width", 0.0).asDouble(),
            size.get("Thickness", 0.0).asDouble()
        };
    }

    // Parse FlipFaceNormal
    if (doorJson.isMember("FlipFaceNormal")) {
        const auto& flipFace = doorJson["FlipFaceNormal"];
        door.FlipFaceNormal = Point{
            flipFace["x"].asDouble(),
            flipFace["y"].asDouble(),
            flipFace["z"].asDouble()
        };
    }

    // Parse FlipHandNormal
    if (doorJson.isMember("FlipHandNormal")) {
        const auto& flipHand = doorJson["FlipHandNormal"];
        door.FlipHandNormal = Point{
            flipHand["x"].asDouble(),
            flipHand["y"].asDouble(),
            flipHand["z"].asDouble()
        };
    }
}


auto parseARDesign(const std::string& arDesignJson) -> ARDesign {
    ARDesign design;
    Json::Value j;
    Json::Reader reader;
    
    if (!reader.parse(arDesignJson, j)) {
        throw std::runtime_error("Failed to parse AR Design JSON");
    }
    
    try {
        // Parse Floor array
        if (j.isMember("Floor") && j["Floor"].isArray()) {
            for (const auto& floorJson : j["Floor"]) {
                Floor floor;
                
                // Add debug output
                // std::cout << "Parsing floor: " << floorJson.toStyledString() << std::endl;
                
                // Parse floor basic info
                floor.Name = floorJson.get("Name", "").asString();
                floor.Num = floorJson.get("Num", "").asString();
                floor.LevelHeight = floorJson.get("LevelHeight", 0.0).asDouble();
                
                // Parse Construction
                if (floorJson.isMember("Construction")) {
                    std::cout << "Found Construction field" << std::endl;
                    const auto& constJson = floorJson["Construction"];
                    
                    // Parse HouseTypes
                    if (constJson.isMember("HouseType")) {
                        std::cout << "Found HouseTypes, count: " << constJson["HouseType"].size() << std::endl;
                    } else {
                        std::cout << "No HouseTypes field found" << std::endl;
                    }
                    
                    // Parse Rooms
                    if (constJson.isMember("Room")) {
                        std::cout << "Found Rooms, count: " << constJson["Room"].size() << std::endl;
                        for (const auto& roomJson : constJson["Room"]) {
                            Room room;
                            parseRoom(roomJson, room);
                            floor.construction.rooms.push_back(room);
                        }
                    } else {
                        std::cout << "No Rooms field found" << std::endl;
                    }
                    
                    // Parse JCWs
                    if (constJson.isMember("JCW")) {
                        const auto& jcws = constJson["JCW"];
                        std::cout << "Found JCWs field, count: " << jcws.size() << std::endl;
                        for (const auto& jcwJson : jcws) {
                            JCW jcw;
                            parseJCW(jcwJson, jcw);
                            floor.construction.jcws.push_back(jcw);
                        }
                    } else {
                        std::cout << "No JCWs field found" << std::endl;
                    }

                    // Parse Doors
                    if (constJson.isMember("Door")) {
                        const auto& doors = constJson["Door"];
                        std::cout << "Found Doors field, count: " << doors.size() << std::endl;
                        for (const auto& doorJson : doors) {
                            Door door;
                            parseDoor(doorJson, door);
                            floor.construction.door.push_back(door);
                        }
                    } else {
                        std::cout << "No Doors field found" << std::endl;
                    }

                    // After parsing all Construction data
                    std::cout << "Construction parsing complete:" << std::endl;
                    std::cout << "  HouseTypes count: " << floor.construction.houseTypes.size() << std::endl;
                    std::cout << "  Rooms count: " << floor.construction.rooms.size() << std::endl;
                    std::cout << "  JCWs count: " << floor.construction.jcws.size() << std::endl;
                    std::cout << "  Doors count: " << floor.construction.door.size() << std::endl;
                } else {
                    std::cout << "No Construction field found" << std::endl;
                }
                design.Floor.push_back(floor);
            }
        }
    } catch (const Json::Exception& e) {
        std::cerr << "JSON parsing error in parseARDesign: " << e.what() << std::endl;
        throw;
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