import copy

from robosuite.models.objects.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import new_joint, array_to_string, xml_path_completion

#class DrawerObject(MujocoXMLObject):
#    def __init__(self):
#        super().__init__('/home/ajinkya/Software/screwNet/data/objects/drawer.xml')

class MicrowaveObject(MujocoXMLObject):
    def __init__(self):
        super().__init__('/home/ajinkya/Software/screwNet/data/objects/microwave.xml')

    def get_collision(self, tag_name='cabinet_bottom'):
        for body in self.worldbody.findall('body'):
            if body.get('name') == tag_name:
                return body
        return None

    def get_actuator(self, jnt_name='bottom_left_hinge'):
        for act in self.actuator.findall('velocity'):
            if act.get('joint') == jnt_name:
                return act

    def get_visual(self, tag_name='cabinet_bottom'):
        def update_contact_tags(body, con_af=1, contype=1, gp_name=0):
            for geom in body.findall('geom'):
                geom.set('conaffinity', str(con_af))
                geom.set('contype', str(contype))
                geom.set('group', str(gp_name))
            return body

        for body in self.worldbody.findall('body'):
            if body.get('name') == tag_name:
                update_contact_tags(body, gp_name=0)
                for b in body.findall('body'):
                    update_contact_tags(b, gp_name=0)

                body_copy = copy.deepcopy(body)
                update_contact_tags(body_copy, gp_name=1)
                for b in body_copy.findall('body'):
                    update_contact_tags(b, gp_name=1)
                self.worldbody.append(body_copy)

                return body
        return None

    def get_assets(self):
        return [x for x in self.asset]