// This file is modified from
// https://gitee.com/ukSir/hmchat2/blob/master/entry/src/main/ets/utils/permissionMananger.ets
import { abilityAccessCtrl, bundleManager, common, Permissions } from '@kit.AbilityKit';

export function allAllowed(permissions: Permissions[]): boolean {
  if (permissions.length == 0) {
    return false;
  }

  const mgr: abilityAccessCtrl.AtManager = abilityAccessCtrl.createAtManager();

  const bundleInfo = bundleManager.getBundleInfoForSelfSync(bundleManager.BundleFlag.GET_BUNDLE_INFO_WITH_APPLICATION);

  let tokenID: number = bundleInfo.appInfo.accessTokenId;

  return permissions.every(permission => abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED ==
  mgr.checkAccessTokenSync(tokenID, permission));
}

export async function requestPermissions(permissions: Permissions[]): Promise<boolean> {
  const mgr: abilityAccessCtrl.AtManager = abilityAccessCtrl.createAtManager();
  const context: Context = getContext() as common.UIAbilityContext;

  const result = await mgr.requestPermissionsFromUser(context, permissions);
  return result.authResults.length > 0 && result.authResults.every(authResults => authResults == 0);
}