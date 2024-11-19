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

// Helper function to parse ARDesign.json
ARDesign parseARDesign(const std::string& arDesignJson) {
    ARDesign arDesign;
    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(arDesignJson, root)) {
        throw std::runtime_error("Failed to parse ARDesign.json");
    }

    // Parse Floors
    if (root.isMember("Floor")) {
        const Json::Value& floorsJson = root["Floor"];
        for (const auto& floorJson : floorsJson) {
            Floor floor;
            floor.Name = floorJson["Name"].asString();
            floor.Num = floorJson["Num"].asString();
            floor.AllFloor = floorJson["AllFloor"].asString();
            floor.LevelHeight = floorJson["LevelHeight"].asDouble();
            floor.LevelElevation = floorJson["LevelElevation"].asDouble();
            floor.RNum = floorJson["RNum"].asString();
            floor.DrawingFrameNo = floorJson["DrawingFrameNo"].asString();

            // Parse BasePoint
            if (floorJson.isMember("BasePoint")) {
                floor.BasePoint = Point{
                    floorJson["BasePoint"]["x"].asDouble(),
                    floorJson["BasePoint"]["y"].asDouble(),
                    floorJson["BasePoint"]["z"].asDouble()
                };
            }

            // Parse Construction
            if (floorJson.isMember("Construction")) {
                const Json::Value& constructionJson = floorJson["Construction"];
                
                // Parse HouseTypes
                if (constructionJson.isMember("HouseType")) {
                    for (const auto& houseTypeJson : constructionJson["HouseType"]) {
                        HouseType houseType;
                        houseType.houseName = houseTypeJson["houseName"].asString();
                        
                        // Parse RoomNames
                        for (const auto& roomName : houseTypeJson["RoomNames"]) {
                            houseType.RoomNames.push_back(roomName.asString());
                        }
                        
                        // Parse Boundary
                        for (const auto& pointJson : houseTypeJson["Boundary"]) {
                            houseType.Boundary.push_back(Point{
                                pointJson["x"].asDouble(),
                                pointJson["y"].asDouble(),
                                pointJson["z"].asDouble()
                            });
                        }
                        floor.construction.houseTypes.push_back(houseType);
                    }
                }

                // Parse Rooms
                if (constructionJson.isMember("Room")) {
                    for (const auto& roomJson : constructionJson["Room"]) {
                        Room room;
                        room.Guid = roomJson["Guid"].asString();
                        room.MapGuid = roomJson["MapGuid"].asString();
                        room.Category = roomJson["Category"].asString();
                        room.Position = roomJson["Position"].asString();
                        room.BlCreateRoom = roomJson["BlCreateRoom"].asInt();
                        room.Name = roomJson["Name"].asString();
                        room.NameType = roomJson["NameType"].asString();
                        room.ShowArea = roomJson["ShowArea"].asDouble();
                        room.LevelOffset = roomJson["LevelOffset"].asDouble();
                        room.LevelOffsetType = roomJson["LevelOffsetType"].asInt();
                        room.ArchThickness = roomJson["ArchThickness"].asDouble();
                        room.STOffSet = roomJson["STOffSet"].asDouble();
                        room.LightArea = roomJson["LightArea"].asDouble();
                        room.AirArea = roomJson["AirArea"].asDouble();
                        room.Area = roomJson["Area"].asDouble();
                        room.RoomElementId = roomJson["RoomElementId"].asInt();
                        room.SectionSymbolName = roomJson["SectionSymbolName"].asString();
                        room.IsOpen = roomJson["IsOpen"].asBool();
                        room.RoomNumber = roomJson["RoomNumber"].asString();
                        room.Number = roomJson["Number"].asInt();

                        // Parse Names array
                        for (const auto& name : roomJson["Names"]) {
                            room.Names.push_back(name.asString());
                        }

                        // Parse DoorIds array
                        for (const auto& id : roomJson["DoorIds"]) {
                            room.DoorIds.push_back(id.asString());
                        }

                        // Parse DoorNums array
                        for (const auto& num : roomJson["DoorNums"]) {
                            room.DoorNums.push_back(num.asString());
                        }

                        // Parse WindowIds array
                        for (const auto& id : roomJson["WindowIds"]) {
                            room.WindowIds.push_back(id.asString());
                        }

                        // Parse DoorAndWindowIds array
                        for (const auto& id : roomJson["DoorAndWindowIds"]) {
                            room.DoorAndWindowIds.push_back(id.asString());
                        }

                        // Parse JCWGuidNames array
                        for (const auto& name : roomJson["JCWGuidNames"]) {
                            room.JCWGuidNames.push_back(name.asString());
                        }

                        // Parse Boundary curves
                        for (const auto& curveJson : roomJson["Boundary"]) {
                            CurveInfo curve;
                            parseCurveInfo(curveJson, curve);
                            room.Boundary.push_back(curve);
                        }

                        // Parse AnnotationPoint
                        if (roomJson.isMember("AnnotationPoint")) {
                            room.AnnotationPoint = Point{
                                roomJson["AnnotationPoint"]["x"].asDouble(),
                                roomJson["AnnotationPoint"]["y"].asDouble(),
                                roomJson["AnnotationPoint"]["z"].asDouble()
                            };
                        }

                        // Parse FloorDrainPoints array
                        for (const auto& pointJson : roomJson["FloorDrainPoints"]) {
                            room.FloorDrainPoints.push_back(Point{
                                pointJson["x"].asDouble(),
                                pointJson["y"].asDouble(),
                                pointJson["z"].asDouble()
                            });
                        }

                        floor.construction.rooms.push_back(room);
                    }
                }

                // Parse JCWs
                if (constructionJson.isMember("JCW")) {
                    for (const auto& jcwJson : constructionJson["JCW"]) {
                        JCW jcw;
                        jcw.GuidName = jcwJson["GuidName"].asString();
                        jcw.Type = jcwJson["Type"].asInt();
                        jcw.Name = jcwJson["Name"].asString();
                        jcw.IsBlockLayer = jcwJson["IsBlockLayer"].asBool();

                        // Parse CenterPoint
                        if (jcwJson.isMember("CenterPoint")) {
                            jcw.CenterPoint = Point{
                                jcwJson["CenterPoint"]["x"].asDouble(),
                                jcwJson["CenterPoint"]["y"].asDouble(),
                                jcwJson["CenterPoint"]["z"].asDouble()
                            };
                        }

                        // Parse ShowCurves
                        for (const auto& curveJson : jcwJson["ShowCurves"]) {
                            CurveInfo curve;
                            parseCurveInfo(curveJson, curve);
                            jcw.ShowCurves.push_back(curve);
                        }

                        // Parse MaxBoundaryCurves
                        for (const auto& curveJson : jcwJson["MaxBoundaryCurves"]) {
                            CurveInfo curve;
                            parseCurveInfo(curveJson, curve);
                            jcw.MaxBoundaryCurves.push_back(curve);
                        }

                        // Parse BoundaryLines
                        for (const auto& curveJson : jcwJson["BoundaryLines"]) {
                            CurveInfo curve;
                            parseCurveInfo(curveJson, curve);
                            jcw.BoundaryLines.push_back(curve);
                        }

                        // Parse Parameters
                        const Json::Value& paramsJson = jcwJson["Parameters"];
                        for (const auto& key : paramsJson.getMemberNames()) {
                            jcw.Parameters[key] = paramsJson[key].asString();
                        }

                        floor.construction.jcws.push_back(jcw);
                    }
                }

                // Parse DoorAndWindows
                if (constructionJson.isMember("DoorAndWindow")) {
                    for (const auto& dwJson : constructionJson["DoorAndWindow"]) {
                        DoorAndWindow dw;
                        dw.Guid = dwJson["Guid"].asString();
                        dw.Type = dwJson["Type"].asInt();
                        dw.Name = dwJson["Name"].asString();
                        dw.NumberForModeling = dwJson["NumberForModeling"].asString();
                        dw.Material = dwJson["Material"].asString();
                        dw.FamilyType = dwJson["FamilyType"].asString();
                        dw.WindowType = dwJson["WindowType"].asString();

                        // Parse Parameters
                        const Json::Value& paramsJson = dwJson["Parameters"];
                        for (const auto& key : paramsJson.getMemberNames()) {
                            dw.Parameters[key] = paramsJson[key].asString();
                        }

                        // Parse WindowFrame
                        for (const auto& curveJson : dwJson["WindowFrame"]) {
                            CurveInfo curve{};
                            parseCurveInfo(curveJson, curve);
                            dw.WindowFrame.push_back(curve);
                        }

                        // Parse WindowGlassModel
                        for (const auto& curveJson : dwJson["WindowGlassModel"]) {
                            CurveInfo curve{};
                            parseCurveInfo(curveJson, curve);
                            dw.WindowGlassModel.push_back(curve);
                        }

                        // Parse OpenInfo
                        const Json::Value& openInfoJson = dwJson["OpenInfo"];
                        for (const auto& key : openInfoJson.getMemberNames()) {
                            dw.OpenInfo[key] = openInfoJson[key].asString();
                        }

                        // Parse WindowWidthLine
                        if (dwJson.isMember("WindowWidthLine")) {
                            parseCurveInfo(dwJson["WindowWidthLine"], dw.WindowWidthLine);
                        }

                        // Parse Items
                        for (const auto& item : dwJson["Items"]) {
                            dw.Items.push_back(item.asString());
                        }

                        floor.construction.doorAndWindows.push_back(dw);
                    }
                }
            }
            
            arDesign.Floor.push_back(floor);
        }
    }

    // Parse ARUniformStanDards
    if (root.isMember("ARUniformStanDards")) {
        const Json::Value& standardsJson = root["ARUniformStanDards"];
        
        // Parse CompanyUnifiedStandarsConfig
        if (standardsJson.isMember("CompanyUnifiedStandarsConfig")) {
            const Json::Value& companyConfig = standardsJson["CompanyUnifiedStandarsConfig"];
            auto& config = arDesign.ARUniformStanDards.CompanyUnifiedStandarsConfig;
            
            // Parse InternalDimensionConfig
            if (companyConfig.isMember("InternalDimensionConfig")) {
                config.InternalDimensionConfig.IsDimension_inHouse = 
                    companyConfig["InternalDimensionConfig"]["IsDimension_inHouse"].asBool();
            }

            // Parse Rail configs
            if (companyConfig.isMember("Rail_balconyConfig")) {
                config.Rail_balconyConfig.Style = companyConfig["Rail_balconyConfig"]["Style"].asString();
                config.Rail_balconyConfig.Location = companyConfig["Rail_balconyConfig"]["Location"].asString();
            }
            if (companyConfig.isMember("Rail_protectWinConfig")) {
                config.Rail_protectWinConfig.Style = companyConfig["Rail_protectWinConfig"]["Style"].asString();
                config.Rail_protectWinConfig.Location = companyConfig["Rail_protectWinConfig"]["Location"].asString();
            }

            // Parse DoorAndWinStyleConfig
            if (companyConfig.isMember("DoorAndWinStyleConfig")) {
                const auto& dwConfig = companyConfig["DoorAndWinStyleConfig"];
                config.DoorAndWinStyleConfig.Style_Elevation = dwConfig["Style_Elevation"].asString();
                config.DoorAndWinStyleConfig.WindowFrameWidth = dwConfig["WindowFrameWidth"].asString();
                config.DoorAndWinStyleConfig.GlassSymbolStyle.IsHasGlassSymbol_FixedWin = 
                    dwConfig["GlassSymbolStyle"]["IsHasGlassSymbol_FixedWin"].asBool();
                config.DoorAndWinStyleConfig.GlassSymbolStyle.IsHasGlassSymbol_OpenWin = 
                    dwConfig["GlassSymbolStyle"]["IsHasGlassSymbol_OpenWin"].asBool();
                config.DoorAndWinStyleConfig.WinOpenLineStyle = dwConfig["WinOpenLineStyle"].asString();
            }

            // Parse PlaneElementsStyleConfig
            if (companyConfig.isMember("PlaneElementsStyleConfig")) {
                const auto& peConfig = companyConfig["PlaneElementsStyleConfig"];
                auto parsePlaneElementStyle = [](const Json::Value& json) {
                    PlaneElementStyle style;
                    style.Type = json["Type"].asString();
                    style.DimensionDrawing = json["DimensionDrawing"].asInt();
                    style.IndexNumbel = json["IndexNumbel"].asString();
                    style.LegendStyle = json["LegendStyle"].asString();
                    return style;
                };

                // Parse all plane element styles
                const std::vector<std::pair<std::string, PlaneElementStyle&>> elements = {
                    {"RainPip", config.PlaneElementsStyleConfig.RainPip},
                    {"CondensatePip", config.PlaneElementsStyleConfig.CondensatePip},
                    {"SewagePip_balcony", config.PlaneElementsStyleConfig.SewagePip_balcony},
                    {"SewagePip_kitAndtoi", config.PlaneElementsStyleConfig.SewagePip_kitAndtoi},
                    {"WastePip", config.PlaneElementsStyleConfig.WastePip},
                    {"FireStandPip", config.PlaneElementsStyleConfig.FireStandPip},
                    {"FireHydrant", config.PlaneElementsStyleConfig.FireHydrant},
                    {"Drain_1", config.PlaneElementsStyleConfig.Drain_1},
                    {"Drain_2", config.PlaneElementsStyleConfig.Drain_2},
                    {"ElectricalBoxHigh_V", config.PlaneElementsStyleConfig.ElectricalBoxHigh_V},
                    {"ElectricalBoxLow_V", config.PlaneElementsStyleConfig.ElectricalBoxLow_V},
                    {"VideoPhone", config.PlaneElementsStyleConfig.VideoPhone},
                    {"RainStrainer_Roof", config.PlaneElementsStyleConfig.RainStrainer_Roof},
                    {"RainStrainer_Side", config.PlaneElementsStyleConfig.RainStrainer_Side},
                    {"OverflowPipe", config.PlaneElementsStyleConfig.OverflowPipe},
                    {"AirConditionHole_Low", config.PlaneElementsStyleConfig.AirConditionHole_Low},
                    {"AirConditionHole_High", config.PlaneElementsStyleConfig.AirConditionHole_High},
                    {"AirConditionHole_Symbol", config.PlaneElementsStyleConfig.AirConditionHole_Symbol},
                    {"ToiletHole", config.PlaneElementsStyleConfig.ToiletHole},
                    {"KitchenHole", config.PlaneElementsStyleConfig.KitchenHole}
                };

                for (const auto& [key, element] : elements) {
                    if (peConfig.isMember(key)) {
                        element = parsePlaneElementStyle(peConfig[key]);
                    }
                }
            }

            // Parse other configs
            if (companyConfig.isMember("FlueHoleConfig")) {
                config.FlueHoleConfig.FlueHoleSymbol = 
                    companyConfig["FlueHoleConfig"]["FlueHoleSymbol"].asString();
            }

            if (companyConfig.isMember("InsulationConfig")) {
                config.InsulationConfig.InsulationStyle = 
                    companyConfig["InsulationConfig"]["InsulationStyle"].asString();
            }

            if (companyConfig.isMember("SteelLadderConfig")) {
                config.SteelLadderConfig.SteelLadderStyle = 
                    companyConfig["SteelLadderConfig"]["SteelLadderStyle"].asString();
            }

            if (companyConfig.isMember("SplashBlockConfig")) {
                config.SplashBlockConfig.SplashBlockStyle = 
                    companyConfig["SplashBlockConfig"]["SplashBlockStyle"].asString();
            }

            if (companyConfig.isMember("AirConditionerBracketConfig")) {
                config.AirConditionerBracketConfig.FloorHeightSyle = 
                    companyConfig["AirConditionerBracketConfig"]["FloorHeightSyle"].asString();
                config.AirConditionerBracketConfig.HafFloorHeightSyle = 
                    companyConfig["AirConditionerBracketConfig"]["HafFloorHeightSyle"].asString();
            }

            if (companyConfig.isMember("LevelStyleConfig")) {
                config.LevelStyleConfig.PlaneLevelStyle = 
                    companyConfig["LevelStyleConfig"]["PlaneLevelStyle"].asString();
                config.LevelStyleConfig.EleLevelStyle = 
                    companyConfig["LevelStyleConfig"]["EleLevelStyle"].asString();
            }
        }

        // Parse ProjectUnifiedStandarsConfig
        if (standardsJson.isMember("ProjectUnifiedStandarsConfig")) {
            const Json::Value& projectConfig = standardsJson["ProjectUnifiedStandarsConfig"];
            auto& config = arDesign.ARUniformStanDards.ProjectUnifiedStandarsConfig;

            // Parse WallSectionGridConfig
            if (projectConfig.isMember("WallSectionGridConfig")) {
                const auto& wsgConfig = projectConfig["WallSectionGridConfig"];
                config.WallSectionGridConfig.IsTrueNum = wsgConfig["IsTrueNum"].asBool();
                config.WallSectionGridConfig.ReplaceSymbol = wsgConfig["ReplaceSymbol"].asString();
            }

            // Parse RoofFlueConfig
            if (projectConfig.isMember("RoofFlueConfig")) {
                const auto& rfConfig = projectConfig["RoofFlueConfig"];
                config.RoofFlueConfig.IsDrawing = rfConfig["IsDrawing"].asBool();
                config.RoofFlueConfig.Height = rfConfig["Height"].asDouble();
            }

            // Parse DoorSillConfig
            if (projectConfig.isMember("DoorSillConfig")) {
                const auto& dsConfig = projectConfig["DoorSillConfig"];
                config.DoorSillConfig.UnderGroundWaterRoom = dsConfig["UnderGroundWaterRoom"].asDouble();
                config.DoorSillConfig.UnderGroundEleRoom = dsConfig["UnderGroundEleRoom"].asDouble();
                config.DoorSillConfig.EquipmentWell = dsConfig["EquipmentWell"].asDouble();
                config.DoorSillConfig.OutRoof = dsConfig["OutRoof"].asDouble();
                config.DoorSillConfig.HeatRoom = dsConfig["HeatRoom"].asDouble();
                config.DoorSillConfig.WaterPumpRoom = dsConfig["WaterPumpRoom"].asDouble();
                config.DoorSillConfig.ElevatorRoom = dsConfig["ElevatorRoom"].asDouble();
            }

            // Parse RefugeIsFireWinConfig
            if (projectConfig.isMember("RefugeIsFireWinConfig")) {
                config.RefugeIsFireWinConfig = projectConfig["RefugeIsFireWinConfig"].asBool();
            }

            // Parse ARInsulationConfig
            if (projectConfig.isMember("ARInsulationConfig")) {
                const auto& insConfig = projectConfig["ARInsulationConfig"];
                config.ARInsulationConfig.Style = insConfig["Style"].asString();
                config.ARInsulationConfig.Materials = insConfig["Materials"].asString();
                config.ARInsulationConfig.Thickness = insConfig["Thickness"].asDouble();
            }
        }
    }

    // Parse other top-level members
    arDesign.ARHeight = root.get("ARHeight", 0.0).asDouble();
    arDesign.FloorNumber = root.get("FloorNumber", 0).asInt();

    // Parse Level
    if (root.isMember("Level")) {
        for (const auto& levelJson : root["Level"]) {
            Level level;
            level.Name = levelJson["Name"].asString();
            level.Elevation = levelJson["Elevation"].asDouble();
            arDesign.Level.push_back(level);
        }
    }

    // Parse Grid
    if (root.isMember("Grid")) {
        for (const auto& gridJson : root["Grid"]) {
            Grid grid;
            grid.GridTextNote = gridJson["GridTextNote"].asString();
            grid.ElementId = gridJson["ElementId"].asInt();
            grid.Coordinate = gridJson["Coordinate"].asString();
            grid.Mode = gridJson["Mode"].asInt();

            // Parse CurveArray
            for (const auto& curveJson : gridJson["CurveArray"]) {
                CurveInfo curve;
                parseCurveInfo(curveJson, curve);
                grid.CurveArray.push_back(curve);
            }

            arDesign.Grid.push_back(grid);
        }
    }

    // Parse StandardInfo
    if (root.isMember("StandardInfo")) {
        for (const auto& standardJson : root["StandardInfo"]) {
            StandardInfo standard;
            standard.AllFloorNums = standardJson["AllFloorNums"].asString();
            standard.Num = standardJson["Num"].asString();
            arDesign.StandardInfo.push_back(standard);
        }
    }

    // Parse WebParam
    if (root.isMember("WebParam")) {
        const auto& webParamJson = root["WebParam"];
    }

    // Parse SectionInfos
    if (root.isMember("SectionInfos")) {
        for (const auto& sectionJson : root["SectionInfos"]) {
            SectionInfo section;

            // Parse Points
            if (sectionJson.isMember("Points")) {
                for (const auto& pointJson : sectionJson["Points"]) {
                }
            }
            
            arDesign.SectionInfos.push_back(section);
        }
    }

    // Parse STSlabs
    if (root.isMember("STSlabs")) {
        for (const auto& slabJson : root["STSlabs"]) {
            STSlab slab;
            slab.Guid = slabJson["Guid"].asString();
            slab.Thickness = slabJson["Thickness"].asDouble();

            // Parse Boundary
            for (const auto& curveJson : slabJson["Boundary"]) {
                CurveInfo curve;
                parseCurveInfo(curveJson, curve);
                slab.Boundary.push_back(curve);
            }
            
            arDesign.STSlabs.push_back(slab);
        }
    }


    return arDesign;
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

