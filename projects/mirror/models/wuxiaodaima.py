# zhushi            #
# values = boxes_cls_pred[:, 4]
#
# # 使用torch.sort()函数对values进行排序，并返回排序后的值和索引
# sorted_values, sorted_indices = torch.sort(values,descending=True)
#
# # 根据排序后的索引对tensor_500_5进行重新排序
# boxes_cls_pred = boxes_cls_pred[sorted_indices]
# boxes_list = []
# #对于置信度大于0.3的目标少于10的图片，将其0.3以下置信度的目标全部剔除
# bj = False
# if boxes_cls_pred[10][4] < 0.3:
#     mask_zhixinduxiaoyuth = boxes_cls_pred[:,4] > 0.3
#     boxes_cls_pred_x = boxes_cls_pred[mask_zhixinduxiaoyuth]
#     boxes_cls_pred = torch.zeros(500, 5, device=boxes_cls_pred.device)
#     boxes_cls_pred[:len(boxes_cls_pred_x)] = boxes_cls_pred_x
#     bj = True
# boxes_cls_pred_clone = boxes_cls_pred.detach()
# boxes_cls_pred_area = box_area(boxes_cls_pred[:,:4])
# area_mean = torch.mean(boxes_cls_pred_area) * 0.5
# score_mean = torch.mean(boxes_cls_pred[:,4])
# box_kiou = box_iou_kk(boxes_cls_pred,boxes_cls_pred_clone)
#
#
# #kiou
# th_1 = 0.3
# #置信度
# th_2 = 0.7
# th_3 = 0.03
# th_4 = 0.9
# th_5 = 0.5
# th_6 = 0.9
# th_7 = 0.5
#
# for i,ki in enumerate(box_kiou):
#     if bj:
#         break
#     sorted_values_1, sorted_indices_1 = torch.sort(ki, descending=True)
#     mask = sorted_values_1 > th_1
#     kiou_da_yu_th = torch.cat([sorted_indices_1[mask][:,None],sorted_values_1[mask][:,None]],1)
#     _,indices = torch.sort(kiou_da_yu_th[:,0])
#     kiou_da_yu_th = kiou_da_yu_th[indices]
#     # boxes_cls_pred_kiou_pai_xu = boxes_cls_pred[sorted_indices_1]
#     boxes_list.append(kiou_da_yu_th)
#
# new_boxes_list = []
# new_boxes_list_res = []
# for b_l in boxes_list:
#     if bj:
#         break
#     if not torch.is_tensor(b_l):
#         continue
#     # boxes_cls_pred[b_l[1:]] = torch.zeros(5).to(boxes_cls_pred.device)
#
#
#
#
#     if boxes_cls_pred[int(b_l[0,0])][4] >= th_2:
#         for bb in b_l:
#             if bb[1] == 1:
#                 continue
#             if boxes_cls_pred[int(bb[0])][4] < 0.6 and bb[1] > th_4:
#                 boxes_cls_pred[int(bb[0])] = torch.zeros(5).to(boxes_cls_pred.device)
#
#     for bb in b_l:
#         if boxes_cls_pred[int(b_l[0, 0])][4] <= score_mean + 0.2 and boxes_cls_pred_area[
#             int(b_l[0, 0])] >= area_mean:
#             boxes_cls_pred[int(bb[0])] = torch.zeros(5).to(boxes_cls_pred.device)
#
#     if boxes_cls_pred[int(b_l[0,0])][4] < th_2 and boxes_cls_pred[int(b_l[0,0])][4] > th_3 and boxes_cls_pred_area[int(b_l[0, 0])] <= area_mean:
#         new_boxes_list.append(b_l[:,0])
#     for b in b_l[:,0]:
#         boxes_list[int(b)] = 0
# # new_boxes_list = []
# if len(new_boxes_list) != 0 and len( new_boxes_list )< 2:
#     xu_yao_jian_ce_s = []
#     xu_yao_jian_ce = []
#     for n_b in new_boxes_list:
#         n_b = torch.round(n_b).long()
#         xu_yao_jian_ce_s.append(boxes_cls_pred_clone[n_b])
#         xu_yao_jian_ce.append(boxes_cls_pred_clone[n_b[0]][:4])
#     zuo_biao,images_list,scale = self.segment_image(batched_inputs[0]['image'],xu_yao_jian_ce)
#     res_chong_xin = []
#     for i,i_l in enumerate(images_list):
#         image_res = []
#         image_dict ={}
#         image_dict['file_name'] = str(i)
#         image_dict['height'] = i_l.shape[1]
#         image_dict['width'] = i_l.shape[2]
#         image_dict['image'] = i_l
#         image_dict['image_id'] = i
#         res_chong_xin_x = self.forward_1([image_dict])
#         # boxes需要处理一下，由原图的坐标映射到分割后的图像
#         boxes = self.yuan2fen(xu_yao_jian_ce_s[i][:,:4],scale,zuo_biao[i])
#         res_chong_xin_x = self.forward_2([image_dict],boxes)
#         res_chong_xin.append(res_chong_xin_x)
#         boxes_cls_pred_1 = self.res_jie_xi_1(res_chong_xin_x)
#         img_path = save_as_img(i_l,str(i))
#         draw_box(img_path,boxes_cls_pred_1,shape=i_l.shape)
#
#
#         # boxes_cls_pred_1[:, [0, 2]] = boxes_cls_pred_1[:, [0, 2]] + xu_yao_jian_ce[i][0]
#         # boxes_cls_pred_1[:, [1, 3]] = boxes_cls_pred_1[:, [1, 3]] + xu_yao_jian_ce[i][1]
#         boxes_cls_pred_1[:,:4] = boxes_cls_pred_1[:,:4] /scale
#         boxes_cls_pred_1[:, [0, 2]] = boxes_cls_pred_1[:, [0, 2]] + zuo_biao[i][0]
#         boxes_cls_pred_1[:, [1, 3]] = boxes_cls_pred_1[:, [1, 3]] + zuo_biao[i][1]
#         chu_shi_kiou = box_iou_kk(xu_yao_jian_ce_s[i][:,:4],boxes_cls_pred_1)
#
#
#         #以kiou排序，如果满足score则选用不满足则赋值为0
#         # for j,cskm in enumerate(chu_shi_kiou):
#         #     chu_shi_kiou_px = self.pai_xu(boxes_cls_pred_1, cskm)
#         #
#         #     dan_ge_res = chu_shi_kiou_px[0]
#         #     image_res.append(dan_ge_res)
#         #     if dan_ge_res[4] < boxes_cls_pred[int(new_boxes_list[i][j])][4] :
#         #     # if dan_ge_res[4] < th_7:
#         #         boxes_cls_pred[int(new_boxes_list[i][j])] = torch.zeros(5).to(boxes_cls_pred.device)
#         #     else:
#         #         boxes_cls_pred[int(new_boxes_list[i][j])] = dan_ge_res
#         #只处理第一个
#         for j,cskm in enumerate(chu_shi_kiou):
#             chu_shi_kiou_px = self.pai_xu(boxes_cls_pred_1, cskm)
#
#             dan_ge_res = chu_shi_kiou_px[0]
#             image_res.append(dan_ge_res)
#             if dan_ge_res[4] < boxes_cls_pred[int(new_boxes_list[i][j])][4] :
#             # if dan_ge_res[4] < th_7:
#                 boxes_cls_pred[int(new_boxes_list[i][j])] = torch.zeros(5).to(boxes_cls_pred.device)
#             else:
#                 boxes_cls_pred[int(new_boxes_list[i][j])] = dan_ge_res
#             break
#             #将与原来预测框kiou大于th_6的框中score最高的加如image_res，且为没有大于th_6的赋值五个零
#         new_boxes_list_res.append(image_res)
#
#         # 方法一挑选kiou大于阈值的box取其中score最大的box
#         # chu_shi_kiou_mask = chu_shi_kiou > th_6
#         # for j,cskm in enumerate(chu_shi_kiou_mask):
#         #     cskk = boxes_cls_pred_1[cskm]
#         #     if len(cskk) == 0:
#         #         dan_ge_res = torch.zeros(5).to(boxes_cls_pred.device)
#         #
#         #
#         #     else:
#         #         dan_ge_res = boxes_cls_pred_1[cskm][0]
#         #     image_res.append(dan_ge_res)
#         #     if dan_ge_res[4] < boxes_cls_pred[int(new_boxes_list[i][j])][4] :
#         #     # if dan_ge_res[4] < th_7:
#         #         boxes_cls_pred[int(new_boxes_list[i][j])] = torch.zeros(5).to(boxes_cls_pred.device)
#         #     else:
#         #         boxes_cls_pred[int(new_boxes_list[i][j])] = dan_ge_res
#         #     #将与原来预测框kiou大于th_6的框中score最高的加如image_res，且为没有大于th_6的赋值五个零
#         # new_boxes_list_res.append(image_res)
#
#
#
#         # self.pai_xu(boxes_cls_pred_1,chu_shi_kiou)
#
#         images_list[i] = image_dict
#
#
#
#
#
#
#
#
#
# # hh = 0
# # for nn in new_boxes_list:
# #    hh = hh + len(nn)
# new_boxes_list_1 = []
# boxes_cls_pred = boxes_cls_pred_clone